// Copyright 2026 @junka
//
// Dispatcher for the vkop op probes (formerly 26 separate executables).
//
//   probes                       list every probe
//   probes <name> [args...]      run one probe; extra args go to the probe
//   probes <name> ... --desc     same, with that probe's descriptor-only env set
//   probes all [--desc]          run every probe, each in a forked child
//
// Each probe runs in a forked child. That is not politeness: several probes
// end in _exit() on purpose (the b3 NVIDIA driver segfaults in its EGL atexit
// handler), and others leave process-wide state behind (VKOP_BDA,
// VKOP_HOST_SHAPE, the shader archive default, the Vulkan instance). A fork
// per probe gives every one of them the fresh process its main() assumed,
// so a sweep is equivalent to running the binaries one by one.

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "probe_common.hpp"

namespace vkop {
namespace probes {

namespace {
std::vector<Probe> &registry() {
    static std::vector<Probe> probes;
    return probes;
}
} // namespace

void register_probe(Probe probe) { registry().push_back(probe); }

const std::vector<Probe> &all_probes() { return registry(); }

namespace {

const Probe *find_probe(const char *name) {
    for (const Probe &p : all_probes()) {
        if (std::strcmp(p.name, name) == 0) {
            return &p;
        }
    }
    return nullptr;
}

void list_probes() {
    std::printf("usage: probes <name> [args...] [--desc] | probes all [--desc]\n\n");
    std::printf("  %-20s %s\n", "all", "run every probe");
    for (const Probe &p : all_probes()) {
        std::printf("  %-20s %s\n", p.name, p.help ? p.help : "");
    }
    std::printf("\nargs after the probe name go to the probe itself, e.g.\n"
                "  probes min-max tiny\n"
                "  probes readback 60 4 200\n"
                "  probes image-layout some/model.vkopbin\n");
}

// Run one probe in a forked child and return its exit code. A signal death
// (the EGL teardown bug) is reported as a failure rather than propagated.
//
// argv[0] is the probe's own name so a probe that printed argv[0] or fed it to
// getopt still sees what its standalone binary saw; the rest is the caller's
// tail. A probe that never touched argv simply ignores the parameters.
int run_forked(const Probe &probe, const std::vector<char *> &args) {
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid < 0) {
        std::perror("fork");
        return 2;
    }
    if (pid == 0) {
        std::vector<char *> child_argv;
        child_argv.reserve(args.size() + 2);
        child_argv.push_back(const_cast<char *>(probe.name));
        for (char *a : args) {
            child_argv.push_back(a);
        }
        child_argv.push_back(nullptr);
        const int rc = probe.run(static_cast<int>(child_argv.size()) - 1,
                                 child_argv.data());
        std::fflush(nullptr);
        std::_Exit(rc);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        std::perror("waitpid");
        return 2;
    }
    if (WIFSIGNALED(status)) {
        std::printf("%s: killed by signal %d\n", probe.name,
                    WTERMSIG(status));
        return 1;
    }
    return WEXITSTATUS(status);
}

} // namespace
} // namespace probes
} // namespace vkop

int main(int argc, char **argv) {
    using namespace vkop::probes;

    bool desc_only = false;
    const char *name = nullptr;
    std::vector<char *> args; // arguments forwarded to the probe unchanged
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--desc") == 0) {
            desc_only = true;
        } else if (!name) {
            name = argv[i];
        } else {
            // Everything after the probe name belongs to the probe: these
            // probes take model paths, counts and mode words (ImageLayoutProbe
            // a .vkopbin, readback_probe n/k/reps, MinMaxProbe full|tiny).
            args.push_back(argv[i]);
        }
    }

    if (!name || std::strcmp(name, "-h") == 0 ||
        std::strcmp(name, "--help") == 0) {
        list_probes();
        return name ? 0 : 2;
    }

    if (std::strcmp(name, "all") == 0) {
        int failed = 0;
        // A sweep runs every probe with no extra arguments: a positional
        // argument means different things to different probes (a model path to
        // image-layout, a mode word to min-max), so there is no useful common
        // meaning to forward. Name the probe to pass it arguments.
        const std::vector<char *> no_args;
        for (const Probe &p : all_probes()) {
            std::printf("\n===== %s =====\n", p.name);
            std::fflush(nullptr);
            // --desc switches on each probe's own *_DESC_ONLY var, so the
            // sweep covers the descriptor fallback too with the same code
            // path the probes have always used.
            if (desc_only && p.desc_env) {
                setenv(p.desc_env, "1", 1);
            }
            const int rc = run_forked(p, no_args);
            if (desc_only && p.desc_env) {
                unsetenv(p.desc_env);
            }
            if (rc != 0) {
                std::printf("---- %s FAILED (rc=%d)\n", p.name, rc);
                failed++;
            }
        }
        std::printf("\n%s: %d probe(s) failed\n",
                    failed ? "FAIL" : "PASS", failed);
        return failed == 0 ? 0 : 1;
    }

    const Probe *probe = find_probe(name);
    if (!probe) {
        std::printf("FAIL: unknown probe '%s'\n\n", name);
        list_probes();
        return 2;
    }
    if (desc_only && probe->desc_env) {
        setenv(probe->desc_env, "1", 1);
    }
    return run_forked(*probe, args);
}
