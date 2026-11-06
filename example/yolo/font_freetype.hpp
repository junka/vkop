// font_freetype.hpp — FreeType-based text renderer with CPU glyph atlas.
//
// Always-on (FreeType is a required build dependency).
// - FreeType renders each glyph as 8-bit grayscale bitmap (FT_LOAD_RENDER).
// - Glyphs are packed into a CPU-side R8 atlas (dynamic row-first packing).
// - drawText() alpha-composites atlas glyphs onto an RGBA pixel buffer.
// - UTF-8 input → codepoint decoding; missing glyph shows '?'.
//
// Threading: FontRenderer is NOT thread-safe. One instance per thread.

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

extern "C" {
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H
}

namespace yolo_font {

// ------------------------------------------------------------------
// UTF-8 → Unicode codepoint iterator
// ------------------------------------------------------------------
struct CodepointIter {
    const char* s;
    const char* end;
    CodepointIter() : s(nullptr), end(nullptr) {}
    CodepointIter(const char* b, const char* e) : s(b), end(e) {}
    bool has() const { return s && s < end; }
    uint32_t next() {
        if (!has()) return 0;
        unsigned char b = *s++;
        if (b < 0x80) return b;
        uint32_t cp = 0; int cont = 0;
        if      ((b & 0xE0) == 0xC0) { cp = b & 0x1F; cont = 1; }
        else if ((b & 0xF0) == 0xE0) { cp = b & 0x0F; cont = 2; }
        else if ((b & 0xF8) == 0xF0) { cp = b & 0x07; cont = 3; }
        else { return 0xFFFD; }
        for (int i = 0; i < cont && s < end; ++i) {
            unsigned char c = *s++;
            if ((c & 0xC0) != 0x80) return 0xFFFD;
            cp = (cp << 6) | (c & 0x3F);
        }
        return cp;
    }
};

// ------------------------------------------------------------------
// Atlas entry — one packed glyph
// ------------------------------------------------------------------
struct Glyph {
    uint32_t codepoint = 0;
    int atlas_x = 0, atlas_y = 0;
    int width = 0, height = 0;
    int bearing_x = 0;
    int bearing_y = 0;
    int advance = 0;
};

// ------------------------------------------------------------------
// FontRenderer
// ------------------------------------------------------------------
class FontRenderer {
public:
    FontRenderer() = default;
    ~FontRenderer() { shutdown(); }

    // Initialize FreeType + load face at the given pixel height.
    // Returns false on any error (FT_Init_FreeType / FT_New_Face /
    // FT_Set_Pixel_Sizes). Callers MUST check the result before drawing.
    bool init(const std::string& font_path, int pixel_height = 24,
              int atlas_w = 512, int atlas_h = 512) {
        shutdown();
        pixel_height_ = pixel_height;
        atlas_w_ = atlas_w;
        atlas_h_ = atlas_h;
        atlas_.assign((size_t)atlas_w * atlas_h, 0);

        if (FT_Init_FreeType(&ft_)) {
            fprintf(stderr, "[font] FT_Init_FreeType failed\n");
            return false;
        }
        if (FT_New_Face(ft_, font_path.c_str(), 0, &face_)) {
            fprintf(stderr, "[font] FT_New_Face failed for '%s'\n", font_path.c_str());
            FT_Done_FreeType(ft_); ft_ = nullptr;
            return false;
        }
        if (FT_Set_Pixel_Sizes(face_, 0, pixel_height)) {
            fprintf(stderr, "[font] FT_Set_Pixel_Sizes failed\n");
            FT_Done_Face(face_); face_ = nullptr;
            FT_Done_FreeType(ft_); ft_ = nullptr;
            return false;
        }
        fprintf(stderr, "[font] loaded '%s' size=%dpx glyphs=%d\n",
                font_path.c_str(), pixel_height, (int)face_->num_glyphs);
        return true;
    }

    bool valid() const { return ft_ != nullptr && face_ != nullptr; }

    // Change pixel height; clears atlas so glyphs re-rasterize next time.
    void setPixelHeight(int h) {
        if (!valid() || h == pixel_height_) return;
        pixel_height_ = h;
        FT_Set_Pixel_Sizes(face_, 0, h);
        glyphs_.clear();
        std::fill(atlas_.begin(), atlas_.end(), 0);
        atlas_cur_x_ = 0; atlas_cur_y_ = 0; atlas_row_h_ = 0;
    }

    int pixelHeight() const { return pixel_height_; }

    int lineHeight() const {
        if (!valid()) return pixel_height_;
        return (int)((face_->size->metrics.ascender - face_->size->metrics.descender) >> 6);
    }

    // Render glyph and optionally composite it into the RGBA buffer.
    // If dst is null, rasterizes and caches only — returns advance.
    int drawGlyph(uint8_t* dst, int W, int H,
                  uint32_t cp, int pen_x, int pen_y,
                  uint8_t fr, uint8_t fg, uint8_t fb, uint8_t fa = 255) {
        auto it = glyphs_.find(cp);
        if (it == glyphs_.end()) {
            if (!rasterize(cp)) {
                it = glyphs_.find('?');
                if (it == glyphs_.end()) return 0;
            } else {
                it = glyphs_.find(cp);
                if (it == glyphs_.end()) return 0;
            }
        }
        const Glyph& g = it->second;

        if (!dst || W <= 0 || H <= 0) return g.advance;

        int sx = pen_x + g.bearing_x;
        int sy = pen_y - g.bearing_y;
        int ex = sx + g.width - 1;
        int ey = sy + g.height - 1;
        if (sx >= W || sy >= H || ex < 0 || ey < 0 || g.width == 0 || g.height == 0)
            return g.advance;

        for (int gy = 0; gy < g.height; ++gy) {
            int dy = sy + gy;
            if (dy < 0 || dy >= H) continue;
            auto* row = dst + dy * W * 4;
            int a_row = (g.atlas_y + gy) * atlas_w_;
            for (int gx = 0; gx < g.width; ++gx) {
                int dx = sx + gx;
                if (dx < 0 || dx >= W) continue;
                uint8_t a_val = atlas_[a_row + g.atlas_x + gx];
                if (a_val == 0) continue;
                uint8_t a = (uint8_t)((uint16_t)a_val * fa / 255);
                uint8_t inv = 255 - a;
                auto* p = row + dx * 4;
                if (a == 255) {
                    p[0] = fr; p[1] = fg; p[2] = fb; p[3] = 255;
                } else {
                    p[0] = (uint8_t)(((uint16_t)fr * a + (uint16_t)p[0] * inv) >> 8);
                    p[1] = (uint8_t)(((uint16_t)fg * a + (uint16_t)p[1] * inv) >> 8);
                    p[2] = (uint8_t)(((uint16_t)fb * a + (uint16_t)p[2] * inv) >> 8);
                    p[3] = 255;
                }
            }
        }
        return g.advance;
    }

    // Render UTF-8 label at pen_x/baseline_y; baseline_y is where glyph
    // baseline sits (so tops of glyphs are above that line by bearing_y).
    // Returns total horizontal advance in pixels.
    int drawText(uint8_t* dst, int W, int H,
                 const std::string& text,
                 int pen_x, int baseline_y,
                 uint8_t fr, uint8_t fg, uint8_t fb, uint8_t fa = 255) {
        int total_w = 0;
        CodepointIter it(text.data(), text.data() + text.size());
        while (it.has()) {
            uint32_t cp = it.next();
            total_w += drawGlyph(dst, W, H, cp, pen_x + total_w, baseline_y,
                                 fr, fg, fb, fa);
        }
        return total_w;
    }

private:
    bool rasterize(uint32_t cp) {
        if (!valid()) return false;
        FT_UInt idx = FT_Get_Char_Index(face_, cp);
        if (idx == 0 && cp != 0x3F) return false;
        if (FT_Load_Glyph(face_, idx, FT_LOAD_RENDER)) return false;
        auto& bm = face_->glyph->bitmap;
        if (bm.width == 0 || bm.rows == 0) return false;
        if (bm.pitch != (int)bm.width) {
            FT_Load_Glyph(face_, idx, FT_LOAD_RENDER | FT_LOAD_TARGET_MONO);
            return false;
        }

        Glyph g;
        g.codepoint = cp;
        g.width = (int)bm.width;
        g.height = (int)bm.rows;
        g.bearing_x = face_->glyph->bitmap_left;
        g.bearing_y = face_->glyph->bitmap_top;
        g.advance = (int)(face_->glyph->advance.x >> 6);

        if (atlas_cur_x_ + g.width > atlas_w_) {
            atlas_cur_x_ = 0;
            atlas_cur_y_ += atlas_row_h_;
            atlas_row_h_ = 0;
        }
        if (atlas_cur_y_ + g.height > atlas_h_) {
            if (atlas_w_ < 1024 && atlas_h_ < 1024) {
                atlas_w_ *= 2; atlas_h_ *= 2;
                atlas_.assign((size_t)atlas_w_ * atlas_h_, 0);
                atlas_cur_x_ = 0; atlas_cur_y_ = 0; atlas_row_h_ = 0;
                glyphs_.clear();  // force all glyphs to re-rasterize next call
                return false;     // callers will re-rasterize cp on next access
            }
            return false;
        }
        g.atlas_x = atlas_cur_x_;
        g.atlas_y = atlas_cur_y_;

        for (int row = 0; row < g.height; ++row) {
            std::memcpy(atlas_.data() + (g.atlas_y + row) * atlas_w_ + g.atlas_x,
                        bm.buffer + row * bm.pitch, g.width);
        }
        atlas_cur_x_ += g.width;
        atlas_row_h_ = std::max(atlas_row_h_, g.height);

        glyphs_[cp] = g;
        return true;
    }

    void shutdown() {
        if (face_) { FT_Done_Face(face_); face_ = nullptr; }
        if (ft_)   { FT_Done_FreeType(ft_); ft_ = nullptr; }
        glyphs_.clear();
        atlas_.clear();
    }

    FT_Library ft_ = nullptr;
    FT_Face    face_ = nullptr;
    int pixel_height_ = 0;

    int atlas_w_ = 0, atlas_h_ = 0;
    std::vector<uint8_t> atlas_;
    int atlas_cur_x_ = 0, atlas_cur_y_ = 0, atlas_row_h_ = 0;

    std::unordered_map<uint32_t, Glyph> glyphs_;
};

} // namespace yolo_font
