// SPIR-V -> LLVM IR -> PPU object -> CUmodule, with caching.
//
// Design notes (why this shape)
// -----------------------------
// * The emitter runs in-process. It is a pure function of the SPIR-V words,
//   needs no external files, and takes tens of microseconds even for the
//   largest shader in the corpus (matmul_down_act, 147 KiB of SPIR-V). A
//   process per shader would add fork+exec+dynamic-link latency for nothing,
//   and would also make fail-loud reporting harder (the reason would arrive as
//   a string on a pipe instead of a return value).
//
// * llc is spawned, not linked. It IS the PPU backend (an 84 MB LLVM build)
//   and is only distributed as a tool. We never use system(): the argument
//   vector is built here, the child is spawned with posix_spawnp (which is
//   what makes this safe from a multi-threaded host -- between fork and exec
//   only async-signal-safe work happens), its status is the real exit code,
//   and stderr is captured through a pipe so a lowering failure surfaces with
//   llc's own diagnostic.
//
// * Everything is cached, twice:
//     - in memory: (SPIR-V bytes, entry name) -> compiled CUmodule, reference
//       counted. Repeat vkCreateComputePipelines with the same shader costs a
//       hash, not a compiler run.
//     - on disk: the object file, keyed by the same hash, so the *next* vkop
//       process reuses it too.
//   The in-memory cache stores the key material itself and compares it, so a
//   hash collision is a miss, never a wrong module.
//
// Failure is always loud: an emitter rejection is reported with the exact
// unsupported instruction, and an llc rejection with its stderr. Nothing here
// falls back to a plausible-but-wrong module.
#include "spirv_compile.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "spirv-emit/emit.hpp"

extern char **environ;

namespace ppu {
namespace {

// ---------------------------------------------------------------------------
// Hashing and cache storage
// ---------------------------------------------------------------------------

// 64-bit FNV-1a. Used only as an index; every hit is verified byte-for-byte.
uint64_t fnv1a(const void *data, size_t bytes, uint64_t seed) {
  const auto *p = static_cast<const uint8_t *>(data);
  uint64_t h = seed;
  for (size_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

// A module is context-scoped: CUmodule handles are only valid in the context
// that loaded them, and vkDestroyDevice destroys that context. The key
// material is kept so a collision can be rejected instead of served.
struct CacheEntry {
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  int refs = 0;
  std::vector<uint32_t> spirv;
  std::string entry;
  CompiledKernel proto;  // module / args / local size / entry name
};

struct Cache {
  std::mutex mu;
  std::map<uint64_t, std::vector<CacheEntry>> buckets;
};

Cache &cache() {
  // Deliberately leaked: CUmodule handles must outlive static destruction (and
  // at process exit there is no useful context to unload them from anyway).
  static Cache *c = new Cache();
  return *c;
}

CacheEntry *find_entry_locked(Cache &c, uint64_t key, CUcontext ctx,
                              const std::vector<uint32_t> &spirv,
                              const std::string &entry) {
  auto b = c.buckets.find(key);
  if (b == c.buckets.end()) return nullptr;
  for (CacheEntry &e : b->second) {
    if (e.context == ctx && e.entry == entry && e.spirv.size() == spirv.size() &&
        std::memcmp(e.spirv.data(), spirv.data(), spirv.size() * sizeof(uint32_t)) == 0) {
      return &e;
    }
  }
  return nullptr;
}

// Object-file cache directory: $VK_PPU_CACHE_DIR, else
// $XDG_CACHE_HOME/vk_ppu_icd, else $HOME/.cache/vk_ppu_icd, else
// /tmp/vk_ppu_icd-cache-$UID. Empty string when none could be created -- the
// cache is an optimisation, never a correctness requirement, so a read-only
// home must not be fatal.
std::string object_cache_dir() {
  static std::once_flag once;
  static std::string dir;
  std::call_once(once, [] {
    std::vector<std::string> candidates;
    if (const char *e = std::getenv("VK_PPU_CACHE_DIR")) candidates.emplace_back(e);
    if (const char *e = std::getenv("XDG_CACHE_HOME"))
      candidates.push_back(std::string(e) + "/vk_ppu_icd");
    if (const char *e = std::getenv("HOME"))
      candidates.push_back(std::string(e) + "/.cache/vk_ppu_icd");
    candidates.push_back("/tmp/vk_ppu_icd-cache-" +
                         std::to_string(static_cast<long>(::getuid())));

    for (const std::string &c : candidates) {
      if (c.empty()) continue;
      std::string partial;
      bool ok = true;
      for (size_t i = 1; i <= c.size(); ++i) {
        if (i == c.size() || c[i] == '/') {
          partial = c.substr(0, i);
          if (::mkdir(partial.c_str(), 0700) != 0 && errno != EEXIST) {
            ok = false;
            break;
          }
        }
      }
      if (ok && ::access(c.c_str(), W_OK) == 0) {
        dir = c;
        return;
      }
    }
    dir.clear();
  });
  return dir;
}

std::string cache_path(const std::string &dir, uint64_t key) {
  char name[40];
  std::snprintf(name, sizeof name, "/%016llx.o", static_cast<unsigned long long>(key));
  return dir + name;
}

bool read_whole_file(const std::string &path, std::vector<uint8_t> &out) {
  std::FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n <= 0) {
    std::fclose(f);
    return false;
  }
  out.resize(static_cast<size_t>(n));
  const size_t got = std::fread(out.data(), 1, out.size(), f);
  std::fclose(f);
  return got == out.size();
}

// ---------------------------------------------------------------------------
// llc discovery
// ---------------------------------------------------------------------------
bool executable_file(const std::string &path) {
  struct stat st{};
  if (::stat(path.c_str(), &st) != 0) return false;
  return S_ISREG(st.st_mode) && ::access(path.c_str(), X_OK) == 0;
}

// ---------------------------------------------------------------------------
// Spawn of llc
// ---------------------------------------------------------------------------

// Run `llc -march=ppu -filetype=obj -o <obj> -`, feeding `ir` on stdin.
//
// The child gets no shell and no injected environment: argv is built here and
// the real environment is passed through. stderr is a pipe, drained
// concurrently with the stdin write -- the corpus's largest IR is 460 KiB,
// larger than a pipe buffer, so writing it all first and reading stderr
// afterwards would deadlock if llc complained early.
bool run_llc(const std::string &llc, const std::string &ir, const std::string &obj_path,
             std::string &error) {
  int in_pipe[2] = {-1, -1};
  int err_pipe[2] = {-1, -1};
  if (::pipe(in_pipe) != 0) {
    error = std::string("pipe() failed: ") + std::strerror(errno);
    return false;
  }
  if (::pipe(err_pipe) != 0) {
    error = std::string("pipe() failed: ") + std::strerror(errno);
    ::close(in_pipe[0]);
    ::close(in_pipe[1]);
    return false;
  }

  std::vector<std::string> argv_store = {llc, "-march=ppu", "-filetype=obj",
                                         "-o", obj_path, "-"};
  std::vector<char *> argv;
  argv.reserve(argv_store.size() + 1);
  for (std::string &s : argv_store) argv.push_back(const_cast<char *>(s.c_str()));
  argv.push_back(nullptr);

  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, in_pipe[0], STDIN_FILENO);
  posix_spawn_file_actions_adddup2(&actions, err_pipe[1], STDERR_FILENO);
  // llc writes the object to -o; anything on stdout is noise that must not
  // interleave with the application's own output.
  posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
  posix_spawn_file_actions_addclose(&actions, in_pipe[1]);
  posix_spawn_file_actions_addclose(&actions, err_pipe[0]);

  pid_t pid = -1;
  const int spawn_rc =
      ::posix_spawnp(&pid, llc.c_str(), &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  ::close(in_pipe[0]);
  ::close(err_pipe[1]);
  if (spawn_rc != 0) {
    ::close(in_pipe[1]);
    ::close(err_pipe[0]);
    error = "cannot run '" + llc + "': " + std::strerror(spawn_rc);
    return false;
  }

  const int in_fd = in_pipe[1];
  const int err_fd = err_pipe[0];
  ::fcntl(in_fd, F_SETFL, ::fcntl(in_fd, F_GETFL) | O_NONBLOCK);
  ::fcntl(err_fd, F_SETFL, ::fcntl(err_fd, F_GETFL) | O_NONBLOCK);

  std::string stderr_text;
  std::string io_error;
  size_t written = 0;
  bool stdin_open = true;
  bool stderr_open = true;

  while (stdin_open || stderr_open) {
    struct pollfd fds[2];
    int n = 0;
    int in_idx = -1, err_idx = -1;
    if (stdin_open) {
      in_idx = n;
      fds[n++] = {in_fd, POLLOUT, 0};
    }
    if (stderr_open) {
      err_idx = n;
      fds[n++] = {err_fd, POLLIN, 0};
    }
    if (n == 0) break;

    const int pr = ::poll(fds, static_cast<nfds_t>(n), 120000);
    if (pr < 0) {
      if (errno == EINTR) continue;
      io_error = std::string("poll() failed: ") + std::strerror(errno);
      break;
    }
    if (pr == 0) {
      io_error = "llc produced no output for 120 s";
      break;
    }
    if (stderr_open && err_idx >= 0 && (fds[err_idx].revents & (POLLIN | POLLHUP | POLLERR))) {
      char buf[8192];
      const ssize_t got = ::read(err_fd, buf, sizeof buf);
      if (got > 0) {
        stderr_text.append(buf, static_cast<size_t>(got));
      } else if (got == 0) {
        stderr_open = false;
      } else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
        stderr_open = false;
      }
    }
    if (stdin_open && in_idx >= 0 && (fds[in_idx].revents & (POLLOUT | POLLERR | POLLHUP))) {
      const ssize_t put = ::write(in_fd, ir.data() + written, ir.size() - written);
      if (put > 0) {
        written += static_cast<size_t>(put);
        if (written == ir.size()) {
          ::close(in_fd);
          stdin_open = false;
        }
      } else if (put < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
        // EPIPE: llc died early. Its stderr (or the exit status) explains why.
        ::close(in_fd);
        stdin_open = false;
      }
    }
  }
  if (stdin_open) ::close(in_fd);
  if (stderr_open) ::close(err_fd);

  if (!io_error.empty()) {
    ::kill(pid, SIGKILL);
    int ignored = 0;
    while (::waitpid(pid, &ignored, 0) < 0 && errno == EINTR) {
    }
    error = io_error;
    if (!stderr_text.empty()) error += ": " + stderr_text;
    return false;
  }

  int status = 0;
  while (::waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) {
      error = std::string("waitpid() failed: ") + std::strerror(errno);
      return false;
    }
  }
  if (WIFSIGNALED(status) || (WIFEXITED(status) && WEXITSTATUS(status) != 0)) {
    error = WIFSIGNALED(status)
                ? "llc was killed by signal " + std::to_string(WTERMSIG(status))
                : "llc exited with status " + std::to_string(WEXITSTATUS(status));
    // llc's diagnostics are the useful part; keep them verbatim.
    std::string trimmed = stderr_text;
    while (!trimmed.empty() && (trimmed.back() == '\n' || trimmed.back() == '\r'))
      trimmed.pop_back();
    if (!trimmed.empty()) error += ": " + trimmed;
    return false;
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------
const char *llc_path(std::string *why) {
  static std::once_flag once;
  static std::string found;
  static std::string reason;
  std::call_once(once, [] {
    // 1. Explicit override. An override that does not exist is an error rather
    //    than a reason to silently use something else: a typo must be visible.
    if (const char *e = std::getenv("VK_PPU_LLC")) {
      if (*e != '\0') {
        if (executable_file(e)) {
          found = e;
          return;
        }
        reason = std::string("VK_PPU_LLC='") + e + "' is not an executable file";
        return;
      }
    }
    // 2. PATH, exactly as a shell would resolve it.
    if (const char *path = std::getenv("PATH")) {
      const std::string p = path;
      size_t start = 0;
      while (start <= p.size()) {
        size_t end = p.find(':', start);
        if (end == std::string::npos) end = p.size();
        if (end > start) {
          const std::string cand = p.substr(start, end - start) + "/llc";
          if (executable_file(cand)) {
            found = cand;
            return;
          }
        }
        start = end + 1;
      }
    }
    // 3. The documented PPU SDK layout, so an application that never sourced
    //    envsetup.sh still works when the SDK is installed in its usual place.
    static const char *kSdkPaths[] = {
        "/usr/local/PPU_SDK/CUDA_SDK/bin/llc",
        "/usr/local/PPU_SDK/bin/llc",
    };
    for (const char *p : kSdkPaths) {
      if (executable_file(p)) {
        found = p;
        return;
      }
    }
    reason =
        "llc not found: set VK_PPU_LLC, or put the PPU SDK's bin directory on "
        "PATH (source /usr/local/PPU_SDK/envsetup.sh)";
  });
  if (why) *why = reason;
  return found.empty() ? nullptr : found.c_str();
}

uint64_t spirv_cache_key(const uint32_t *spirv, size_t word_count, const char *entry_point) {
  // Bump kFormatVersion whenever the emitter's codegen changes in a way that
  // must not be served from an existing on-disk cache. The key is the SPIR-V
  // bytes plus this constant, so a shader re-compiled with a FIXED emitter has
  // exactly the same key as the one the buggy emitter cached -- without a bump
  // the fix is invisible and the stale object keeps running.
  //
  //   1  initial
  //   2  OpVectorShuffle, OpCompositeInsert, runtime vector indices in
  //      OpAccessChain, sin/cos quarter-turn scaling
  constexpr uint64_t kFormatVersion = 2;
  uint64_t h = fnv1a(spirv, word_count * sizeof(uint32_t),
                     1469598103934665603ull ^ kFormatVersion);
  if (entry_point != nullptr)
    h = fnv1a(entry_point, std::strlen(entry_point), h);
  return h;
}

CUresult compile_spirv_to_ppu_kernel(CUcontext context, const uint32_t *spirv,
                                     size_t word_count, const char *entry_point,
                                     CompiledKernel *out, std::string *error) {
  if (out == nullptr) return CUDA_ERROR_INVALID_VALUE;
  *out = CompiledKernel{};
  if (spirv == nullptr || word_count == 0) {
    if (error != nullptr) *error = "empty SPIR-V module";
    return CUDA_ERROR_INVALID_VALUE;
  }

  const std::string entry_req =
      (entry_point != nullptr && *entry_point != '\0') ? entry_point : std::string();
  const uint64_t key = spirv_cache_key(spirv, word_count, entry_req.c_str());
  const std::vector<uint32_t> words(spirv, spirv + word_count);

  // ---- 0. in-memory cache ----
  {
    Cache &c = cache();
    std::lock_guard<std::mutex> lock(c.mu);
    if (CacheEntry *e = find_entry_locked(c, key, context, words, entry_req)) {
      e->refs++;
      *out = e->proto;
      PPU_LOG("spirv: module cache hit for '%s' (refs now %d)", e->proto.entry_name.c_str(),
              e->refs);
      return CUDA_SUCCESS;
    }
  }

  // Wall-clock split of a cold compile, so the cost is visible instead of
  // guessed. VK_PPU_DEBUG=1 prints it.
  const auto t_start = std::chrono::steady_clock::now();
  auto ms_since = [&t_start] {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t_start)
        .count();
  };

  // ---- 1. emit ----
  std::string ir;
  std::string emit_error;
  ppu_emit::KernelInfo info;
  const ppu_emit::Status st = ppu_emit::emit_llvm_ir(
      spirv, word_count, entry_req.empty() ? nullptr : entry_req.c_str(), ir, &info,
      emit_error);
  if (st != ppu_emit::Status::Ok) {
    if (error != nullptr) {
      *error = "SPIR-V -> LLVM emission failed (entry '" +
               std::string(entry_req.empty() ? "?" : entry_req) + "'): " + emit_error;
    }
    return st == ppu_emit::Status::Unsupported ? CUDA_ERROR_NOT_SUPPORTED
                                               : CUDA_ERROR_INVALID_VALUE;
  }

  const double emit_ms = ms_since();

  // ---- 2. llc ----
  std::string llc_why;
  const char *llc = llc_path(&llc_why);
  if (llc == nullptr) {
    if (error != nullptr) *error = "entry '" + info.entry_name + "': " + llc_why;
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  std::vector<uint8_t> object;
  bool have_object = false;

  const std::string dir = object_cache_dir();
  const std::string obj_path = dir.empty() ? std::string() : cache_path(dir, key);
  if (!obj_path.empty() && read_whole_file(obj_path, object)) {
    have_object = true;
    PPU_LOG("spirv: object cache hit for '%s' (%zu bytes)", info.entry_name.c_str(),
            object.size());
  }

  if (!have_object) {
    std::string llc_error;
    if (!obj_path.empty()) {
      if (run_llc(llc, ir, obj_path, llc_error)) have_object = read_whole_file(obj_path, object);
    } else {
      // No cache directory: compile into a private temporary directory.
      char tmpl[] = "/tmp/vk_ppu_obj_XXXXXX";
      const int fd = ::mkstemp(tmpl);
      if (fd < 0) {
        if (error != nullptr) *error = "cannot create a temporary object file";
        return CUDA_ERROR_NOT_SUPPORTED;
      }
      ::close(fd);
      ::unlink(tmpl);  // llc creates it; we only wanted a unique name
      if (run_llc(llc, ir, tmpl, llc_error)) have_object = read_whole_file(tmpl, object);
      ::unlink(tmpl);
    }
    if (!have_object) {
      if (error != nullptr) {
        *error = "llc failed for entry '" + info.entry_name + "'" +
                 (llc_error.empty() ? " (no object produced)" : ": " + llc_error);
      }
      return CUDA_ERROR_NOT_SUPPORTED;
    }
  }

  const double llc_ms = ms_since();

  // ---- 3. cuModuleLoadData ----
  Driver &d = driver();
  if (!d.cuModuleLoadData) {
    if (error != nullptr) *error = "libcuda.so.1 has no cuModuleLoadData";
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUmodule module = nullptr;
  const CUresult load_rc = d.cuModuleLoadData(&module, object.data());
  if (load_rc != CUDA_SUCCESS) {
    if (error != nullptr) {
      *error = "cuModuleLoadData rejected the compiled object for entry '" +
               info.entry_name + "': " + cu_strerror(load_rc);
    }
    return load_rc;
  }

  // Resolve the symbol the application will ask for. This is also the check
  // that the emitter's entry naming really carried the SPIR-V name through, so
  // a naming regression fails here instead of producing a pipeline that cannot
  // be dispatched.
  const std::string lookup_name =
      entry_req.empty() ? info.entry_name : entry_req;
  if (d.cuModuleGetFunction) {
    CUfunction fn = nullptr;
    if (d.cuModuleGetFunction(&fn, module, lookup_name.c_str()) != CUDA_SUCCESS) {
      if (error != nullptr) {
        *error = "compiled object has no kernel named '" + lookup_name +
                 "' (emitter produced '" + info.entry_name + "')";
      }
      if (d.cuModuleUnload) d.cuModuleUnload(module);
      return CUDA_ERROR_NOT_FOUND;
    }
  }

  PPU_LOG("spirv: compiled '%s' cold in %.1f ms (emit %.1f, llc %.1f); IR %zu bytes",
          info.entry_name.c_str(), ms_since(), emit_ms, llc_ms - emit_ms, ir.size());

  out->module = module;
  out->entry_name = info.entry_name;
  for (uint32_t i = 0; i < 3; ++i) out->local_size[i] = info.local_size[i];
  for (const ppu_emit::KernelArg &a : info.args) {
    KernelArgBinding b;
    b.set = a.set;
    b.binding = a.binding;
    b.push_constant = a.push_constant;
    out->args.push_back(b);
  }

  {
    Cache &c = cache();
    std::lock_guard<std::mutex> lock(c.mu);
    auto &bucket = c.buckets[key];
    // Re-check: another thread may have compiled the same key concurrently.
    if (CacheEntry *e = find_entry_locked(c, key, context, words, entry_req)) {
      e->refs++;
      if (d.cuModuleUnload) d.cuModuleUnload(module);  // keep the shared one
      *out = e->proto;
      return CUDA_SUCCESS;
    }
    CacheEntry e;
    e.context = context;
    e.module = module;
    e.refs = 1;
    e.spirv = words;
    e.entry = entry_req;
    e.proto = *out;
    bucket.push_back(std::move(e));
  }
  return CUDA_SUCCESS;
}

void release_compiled_kernel(const CompiledKernel &kernel) {
  if (kernel.module == nullptr) return;
  Cache &c = cache();
  std::lock_guard<std::mutex> lock(c.mu);
  for (auto &bucket : c.buckets) {
    for (auto it = bucket.second.begin(); it != bucket.second.end(); ++it) {
      if (it->module != kernel.module) continue;
      if (--it->refs <= 0) {
        Driver &d = driver();
        if (d.cuModuleUnload) d.cuModuleUnload(it->module);
        bucket.second.erase(it);
      }
      return;
    }
  }
  // Not in the cache (should not happen): still unload, never leak.
  Driver &d = driver();
  if (d.cuModuleUnload) d.cuModuleUnload(kernel.module);
}

void purge_spirv_cache(CUcontext context) {
  Cache &c = cache();
  Driver &d = driver();
  std::lock_guard<std::mutex> lock(c.mu);
  for (auto &bucket : c.buckets) {
    for (auto it = bucket.second.begin(); it != bucket.second.end();) {
      if (it->context == context) {
        if (d.cuModuleUnload && it->module) d.cuModuleUnload(it->module);
        it = bucket.second.erase(it);
      } else {
        ++it;
      }
    }
  }
}

}  // namespace ppu
