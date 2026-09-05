/*
    Scan Tailor - Interactive post-processing tool for scanned pages.
    Copyright (C)  Joseph Artsimovich <joseph.artsimovich@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "Transform.h"
#include "Grayscale.h"
#include "GrayImage.h"
#include <QImage>
#include <QRect>
#include <QSizeF>
#include <QPointF>
#include <QPolygonF>
#include <QColor>
#include <QTransform>
#include <QtGlobal>
#include <QDebug>
#include <stdexcept>
#include <algorithm>
#include <stdint.h>
#include <math.h>
#include <assert.h>

namespace imageproc
{

namespace
{

struct XLess {
    bool operator()(QPointF const& lhs, QPointF const& rhs) const
    {
        return lhs.x() < rhs.x();
    }
};

struct YLess {
    bool operator()(QPointF const& lhs, QPointF const& rhs) const
    {
        return lhs.y() < rhs.y();
    }
};

class Gray
{
public:
    Gray() : m_grayLevel(0) {}

    inline void add(uint8_t const gray_level, unsigned const area)
    {
        m_grayLevel += gray_level * area;
    }

    inline uint8_t result(unsigned const total_area) const
    {
        unsigned const half_area = total_area >> 1;
        unsigned const res = (m_grayLevel + half_area) / total_area;
        return static_cast<uint8_t>(res);
    }
private:
    unsigned m_grayLevel;
};

class RGB32
{
public:
    RGB32() : m_red(0), m_green(0), m_blue(0) {}

    inline void add(uint32_t rgb, unsigned const area)
    {
        m_blue += (rgb & 0xFF) * area;
        rgb >>= 8;
        m_green += (rgb & 0xFF) * area;
        rgb >>= 8;
        m_red += (rgb & 0xFF) * area;
    }

    inline uint32_t result(unsigned const total_area) const
    {
        unsigned const half_area = total_area >> 1;
        uint32_t rgb = 0x0000FF00;
        rgb |= (m_red + half_area) / total_area;
        rgb <<= 8;
        rgb |= (m_green + half_area) / total_area;
        rgb <<= 8;
        rgb |= (m_blue + half_area) / total_area;
        return rgb;
    }
private:
    unsigned m_red;
    unsigned m_green;
    unsigned m_blue;
};

class ARGB32
{
public:
    ARGB32() : m_alpha(0), m_red(0), m_green(0), m_blue(0) {}

    inline void add(uint32_t argb, unsigned const area)
    {
        m_blue += (argb & 0xFF) * area;
        argb >>= 8;
        m_green += (argb & 0xFF) * area;
        argb >>= 8;
        m_red += (argb & 0xFF) * area;
        argb >>= 8;
        m_alpha += argb * area;
    }

    inline uint32_t result(unsigned const total_area) const
    {
        unsigned const half_area = total_area >> 1;
        uint32_t argb = (m_alpha + half_area) / total_area;
        argb <<= 8;
        argb |= (m_red + half_area) / total_area;
        argb <<= 8;
        argb |= (m_green + half_area) / total_area;
        argb <<= 8;
        argb |= (m_blue + half_area) / total_area;
        return argb;
    }
private:
    unsigned m_alpha;
    unsigned m_red;
    unsigned m_green;
    unsigned m_blue;
};

static QSizeF calcSrcUnitSize(QTransform const& xform, QSizeF const& min)
{
    // Imagine a rectangle of (0, 0, 1, 1), except we take
    // centers of its edges instead of its vertices.
    QPolygonF dst_poly;
    dst_poly.push_back(QPointF(0.5, 0.0));
    dst_poly.push_back(QPointF(1.0, 0.5));
    dst_poly.push_back(QPointF(0.5, 1.0));
    dst_poly.push_back(QPointF(0.0, 0.5));

    QPolygonF src_poly(xform.map(dst_poly));
    std::sort(src_poly.begin(), src_poly.end(), XLess());
    double const width = src_poly.back().x() - src_poly.front().x();
    std::sort(src_poly.begin(), src_poly.end(), YLess());
    double const height = src_poly.back().y() - src_poly.front().y();

    QSizeF const min32(min * 32.0);
    return QSizeF(
               std::max(min32.width(), qreal(width)),
               std::max(min32.height(), qreal(height))
           );
}

template<typename StorageUnit, typename Mixer>
static void transformGeneric(
    StorageUnit const* const src_data, int const src_stride, QSize const src_size,
    StorageUnit* const dst_data, int const dst_stride, QTransform const& xform,
    QRect const& dst_rect, StorageUnit const outside_color, int const outside_flags,
    QSizeF const& min_mapping_area)
{
    int const sw = src_size.width();
    int const sh = src_size.height();
    int const dw = dst_rect.width();
    int const dh = dst_rect.height();

    QTransform inv_xform;
    inv_xform.translate(dst_rect.x(), dst_rect.y());
    inv_xform *= xform.inverted();
    inv_xform *= QTransform().scale(32.0, 32.0);

    // sx32 = dx*inv_xform.m11() + dy*inv_xform.m21() + inv_xform.dx();
    // sy32 = dy*inv_xform.m22() + dx*inv_xform.m12() + inv_xform.dy();

    QSizeF const src32_unit_size(calcSrcUnitSize(inv_xform, min_mapping_area));
    int const src32_unit_w = std::max<int>(1, qRound(src32_unit_size.width()));
    int const src32_unit_h = std::max<int>(1, qRound(src32_unit_size.height()));

    #pragma omp parallel for schedule(static) shared(inv_xform)
    for (int dy = 0; dy < dh; ++dy) {
        StorageUnit* dst_line = dst_data + dy * dst_stride;
        double const f_dy_center = dy + 0.5;
        double const f_sx32_base = f_dy_center * inv_xform.m21() + inv_xform.dx();
        double const f_sy32_base = f_dy_center * inv_xform.m22() + inv_xform.dy();

        for (int dx = 0; dx < dw; ++dx) {
            double const f_dx_center = dx + 0.5;
            double const f_sx32_center = f_sx32_base + f_dx_center * inv_xform.m11();
            double const f_sy32_center = f_sy32_base + f_dx_center * inv_xform.m12();
            int src32_left = (int)f_sx32_center - (src32_unit_w >> 1);
            int src32_top = (int)f_sy32_center - (src32_unit_h >> 1);
            int src32_right = src32_left + src32_unit_w;
            int src32_bottom = src32_top + src32_unit_h;
            int src_left = src32_left >> 5;
            int src_right = (src32_right - 1) >> 5; // inclusive
            int src_top = src32_top >> 5;
            int src_bottom = (src32_bottom - 1) >> 5; // inclusive
            assert(src_bottom >= src_top);
            assert(src_right >= src_left);

            if (src_bottom < 0 || src_right < 0 || src_left >= sw || src_top >= sh) {
                // Completely outside of src image.
                if (outside_flags & OutsidePixels::COLOR) {
                    dst_line[dx] = outside_color;
                } else {
                    int const src_x = qBound<int>(0, (src_left + src_right) >> 1, sw - 1);
                    int const src_y = qBound<int>(0, (src_top + src_bottom) >> 1, sh - 1);
                    dst_line[dx] = src_data[src_y * src_stride + src_x];
                }
                continue;
            }

            /*
             * Note that (intval / 32) is not the same as (intval >> 5).
             * The former rounds towards zero, while the latter rounds towards
             * negative infinity.
             * Likewise, (intval % 32) is not the same as (intval & 31).
             * The following expression:
             * top_fraction = 32 - (src32_top & 31);
             * works correctly with both positive and negative src32_top.
             */

            unsigned background_area = 0;

            if (src_top < 0) {
                unsigned const top_fraction = 32 - (src32_top & 31);
                unsigned const hor_fraction = src32_right - src32_left;
                background_area += top_fraction * hor_fraction;
                unsigned const full_pixels_ver = -1 - src_top;
                background_area += hor_fraction * (full_pixels_ver << 5);
                src_top = 0;
                src32_top = 0;
            }
            if (src_bottom >= sh) {
                unsigned const bottom_fraction = src32_bottom - (src_bottom << 5);
                unsigned const hor_fraction = src32_right - src32_left;
                background_area += bottom_fraction * hor_fraction;
                unsigned const full_pixels_ver = src_bottom - sh;
                background_area += hor_fraction * (full_pixels_ver << 5);
                src_bottom = sh - 1; // inclusive
                src32_bottom = sh << 5; // exclusive
            }
            if (src_left < 0) {
                unsigned const left_fraction = 32 - (src32_left & 31);
                unsigned const vert_fraction = src32_bottom - src32_top;
                background_area += left_fraction * vert_fraction;
                unsigned const full_pixels_hor = -1 - src_left;
                background_area += vert_fraction * (full_pixels_hor << 5);
                src_left = 0;
                src32_left = 0;
            }
            if (src_right >= sw) {
                unsigned const right_fraction = src32_right - (src_right << 5);
                unsigned const vert_fraction = src32_bottom - src32_top;
                background_area += right_fraction * vert_fraction;
                unsigned const full_pixels_hor = src_right - sw;
                background_area += vert_fraction * (full_pixels_hor << 5);
                src_right = sw - 1; // inclusive
                src32_right = sw << 5; // exclusive
            }
            assert(src_bottom >= src_top);
            assert(src_right >= src_left);

            Mixer mixer;
            if (outside_flags & OutsidePixels::WEAK) {
                background_area = 0;
            } else {
                mixer.add(outside_color, background_area);
            }

            unsigned const left_fraction = 32 - (src32_left & 31);
            unsigned const top_fraction = 32 - (src32_top & 31);
            unsigned const right_fraction = src32_right - (src_right << 5);
            unsigned const bottom_fraction = src32_bottom - (src_bottom << 5);

            assert(left_fraction + right_fraction + (src_right - src_left - 1) * 32 == static_cast<unsigned>(src32_right - src32_left));
            assert(top_fraction + bottom_fraction + (src_bottom - src_top - 1) * 32 == static_cast<unsigned>(src32_bottom - src32_top));

            unsigned const src_area = (src32_bottom - src32_top) * (src32_right - src32_left);
            if (src_area == 0) {
                if ((outside_flags & OutsidePixels::COLOR)) {
                    dst_line[dx] = outside_color;
                } else {
                    int const src_x = qBound<int>(0, (src_left + src_right) >> 1, sw - 1);
                    int const src_y = qBound<int>(0, (src_top + src_bottom) >> 1, sh - 1);
                    dst_line[dx] = src_data[src_y * src_stride + src_x];
                }
                continue;
            }

            StorageUnit const* src_line = &src_data[src_top * src_stride];

            if (src_top == src_bottom) {
                if (src_left == src_right) {
                    // dst pixel maps to a single src pixel
                    StorageUnit const c = src_line[src_left];
                    if (background_area == 0) {
                        // common case optimization
                        dst_line[dx] = c;
                        continue;
                    }
                    mixer.add(c, src_area);
                } else {
                    // dst pixel maps to a horizontal line of src pixels
                    unsigned const vert_fraction = src32_bottom - src32_top;
                    unsigned const left_area = vert_fraction * left_fraction;
                    unsigned const middle_area = vert_fraction << 5;
                    unsigned const right_area = vert_fraction * right_fraction;

                    mixer.add(src_line[src_left], left_area);

                    for (int sx = src_left + 1; sx < src_right; ++sx) {
                        mixer.add(src_line[sx], middle_area);
                    }

                    mixer.add(src_line[src_right], right_area);
                }
            } else if (src_left == src_right) {
                // dst pixel maps to a vertical line of src pixels
                unsigned const hor_fraction = src32_right - src32_left;
                unsigned const top_area = hor_fraction * top_fraction;
                unsigned const middle_area = hor_fraction << 5;
                unsigned const bottom_area =  hor_fraction * bottom_fraction;

                src_line += src_left;
                mixer.add(*src_line, top_area);

                src_line += src_stride;

                for (int sy = src_top + 1; sy < src_bottom; ++sy) {
                    mixer.add(*src_line, middle_area);
                    src_line += src_stride;
                }

                mixer.add(*src_line, bottom_area);
            } else {
                // dst pixel maps to a block of src pixels
                unsigned const top_area = top_fraction << 5;
                unsigned const bottom_area = bottom_fraction << 5;
                unsigned const left_area = left_fraction << 5;
                unsigned const right_area = right_fraction << 5;
                unsigned const topleft_area = top_fraction * left_fraction;
                unsigned const topright_area = top_fraction * right_fraction;
                unsigned const bottomleft_area = bottom_fraction * left_fraction;
                unsigned const bottomright_area = bottom_fraction * right_fraction;

                // process the top-left corner
                mixer.add(src_line[src_left], topleft_area);

                // process the top line (without corners)
                for (int sx = src_left + 1; sx < src_right; ++sx) {
                    mixer.add(src_line[sx], top_area);
                }

                // process the top-right corner
                mixer.add(src_line[src_right], topright_area);

                src_line += src_stride;

                // process middle lines
                for (int sy = src_top + 1; sy < src_bottom; ++sy) {
                    mixer.add(src_line[src_left], left_area);

                    for (int sx = src_left + 1; sx < src_right; ++sx) {
                        mixer.add(src_line[sx], 32 * 32);
                    }

                    mixer.add(src_line[src_right], right_area);

                    src_line += src_stride;
                }

                // process bottom-left corner
                mixer.add(src_line[src_left], bottomleft_area);

                // process the bottom line (without corners)
                for (int sx = src_left + 1; sx < src_right; ++sx) {
                    mixer.add(src_line[sx], bottom_area);
                }

                // process the bottom-right corner
                mixer.add(src_line[src_right], bottomright_area);
            }

            dst_line[dx] = mixer.result(src_area + background_area);
        }
    }
}

struct CatmullRomWeights {
    inline void operator()(float u, float& w_m1, float& w_0, float& w_1, float& w_2) const
    {
        float const u2 = u * u;
        float const u3 = u2 * u;
        w_m1 = -0.5f * u3 + u2 - 0.5f * u;
        w_0  =  1.5f * u3 - 2.5f * u2 + 1.0f;
        w_1  = -1.5f * u3 + 2.0f * u2 + 0.5f * u;
        w_2  =  0.5f * u3 - 0.5f * u2;
    }
};

struct MitchellWeights {
    inline void operator()(float u, float& w_m1, float& w_0, float& w_1, float& w_2) const
    {
        float const u2 = u * u;
        float const u3 = u2 * u;
        float const inv18 = 1.0f / 18.0f;
        w_m1 = (-7.0f * u3 + 15.0f * u2 - 9.0f * u + 1.0f) * inv18;
        w_0  = ( 21.0f * u3 - 36.0f * u2 + 16.0f) * inv18;
        w_1  = (-21.0f * u3 + 27.0f * u2 + 9.0f * u + 1.0f) * inv18;
        w_2  = (  7.0f * u3 -  6.0f * u2) * inv18;
    }
};

template<typename WeightCalculator>
static void transformBicubicGray(
    uint8_t const* const src_data, int const src_stride, QSize const src_size,
    uint8_t* const dst_data, int const dst_stride,
    QTransform const& inv_xform, QRect const& dst_rect,
    uint8_t const outside_color, int const outside_flags)
{
    int const sw = src_size.width();
    int const sh = src_size.height();
    int const dw = dst_rect.width();
    int const dh = dst_rect.height();
    WeightCalculator calc_weights;

    #pragma omp parallel for schedule(static) shared(inv_xform)
    for (int dy = 0; dy < dh; ++dy) {
        uint8_t* dst_line = dst_data + dy * dst_stride;
        double const f_dy_center = dy + 0.5;
        double const f_sx_base = f_dy_center * inv_xform.m21() + inv_xform.dx();
        double const f_sy_base = f_dy_center * inv_xform.m22() + inv_xform.dy();

        for (int dx = 0; dx < dw; ++dx) {
            double const f_dx_center = dx + 0.5;
            double const sx = f_sx_base + f_dx_center * inv_xform.m11();
            double const sy = f_sy_base + f_dx_center * inv_xform.m12();

            double const x_prime = sx - 0.5;
            double const y_prime = sy - 0.5;

            int const x0 = (int)std::floor(x_prime);
            int const y0 = (int)std::floor(y_prime);

            if (x0 < -2 || x0 >= sw + 1 || y0 < -2 || y0 >= sh + 1) {
                if (outside_flags & OutsidePixels::COLOR) {
                    dst_line[dx] = outside_color;
                } else {
                    int const cx = qBound(0, x0, sw - 1);
                    int const cy = qBound(0, y0, sh - 1);
                    dst_line[dx] = src_data[cy * src_stride + cx];
                }
                continue;
            }

            float const u = (float)(x_prime - x0);
            float const v = (float)(y_prime - y0);

            float wx_m1, wx_0, wx_1, wx_2;
            calc_weights(u, wx_m1, wx_0, wx_1, wx_2);

            float wy_m1, wy_0, wy_1, wy_2;
            calc_weights(v, wy_m1, wy_0, wy_1, wy_2);

            if (x0 >= 1 && x0 + 2 < sw && y0 >= 1 && y0 + 2 < sh) {
                uint8_t const* p_row_m1 = src_data + (y0 - 1) * src_stride + (x0 - 1);
                uint8_t const* p_row_0  = p_row_m1 + src_stride;
                uint8_t const* p_row_1  = p_row_0 + src_stride;
                uint8_t const* p_row_2  = p_row_1 + src_stride;

                float const r_m1 = wx_m1 * p_row_m1[0] + wx_0 * p_row_m1[1] + wx_1 * p_row_m1[2] + wx_2 * p_row_m1[3];
                float const r_0  = wx_m1 * p_row_0[0]  + wx_0 * p_row_0[1]  + wx_1 * p_row_0[2]  + wx_2 * p_row_0[3];
                float const r_1  = wx_m1 * p_row_1[0]  + wx_0 * p_row_1[1]  + wx_1 * p_row_1[2]  + wx_2 * p_row_1[3];
                float const r_2  = wx_m1 * p_row_2[0]  + wx_0 * p_row_2[1]  + wx_1 * p_row_2[2]  + wx_2 * p_row_2[3];

                float const val = wy_m1 * r_m1 + wy_0 * r_0 + wy_1 * r_1 + wy_2 * r_2;
                int const ival = (int)std::round(val);
                dst_line[dx] = (uint8_t)qBound(0, ival, 255);
            } else {
                auto fetch = [&](int const x, int const y) -> float {
                    if (x >= 0 && x < sw && y >= 0 && y < sh) {
                        return (float)src_data[y * src_stride + x];
                    }
                    if (outside_flags & OutsidePixels::COLOR) {
                        return (float)outside_color;
                    }
                    int const cx = qBound(0, x, sw - 1);
                    int const cy = qBound(0, y, sh - 1);
                    return (float)src_data[cy * src_stride + cx];
                };

                float const r_m1 = wx_m1 * fetch(x0 - 1, y0 - 1) + wx_0 * fetch(x0, y0 - 1) + wx_1 * fetch(x0 + 1, y0 - 1) + wx_2 * fetch(x0 + 2, y0 - 1);
                float const r_0  = wx_m1 * fetch(x0 - 1, y0)     + wx_0 * fetch(x0, y0)     + wx_1 * fetch(x0 + 1, y0)     + wx_2 * fetch(x0 + 2, y0);
                float const r_1  = wx_m1 * fetch(x0 - 1, y0 + 1) + wx_0 * fetch(x0, y0 + 1) + wx_1 * fetch(x0 + 1, y0 + 1) + wx_2 * fetch(x0 + 2, y0 + 1);
                float const r_2  = wx_m1 * fetch(x0 - 1, y0 + 2) + wx_0 * fetch(x0, y0 + 2) + wx_1 * fetch(x0 + 1, y0 + 2) + wx_2 * fetch(x0 + 2, y0 + 2);

                float const val = wy_m1 * r_m1 + wy_0 * r_0 + wy_1 * r_1 + wy_2 * r_2;
                int const ival = (int)std::round(val);
                dst_line[dx] = (uint8_t)qBound(0, ival, 255);
            }
        }
    }
}

template<typename WeightCalculator>
static void transformBicubicRGB32(
    uint32_t const* const src_data, int const src_stride, QSize const src_size,
    uint32_t* const dst_data, int const dst_stride,
    QTransform const& inv_xform, QRect const& dst_rect,
    uint32_t const outside_color, int const outside_flags)
{
    int const sw = src_size.width();
    int const sh = src_size.height();
    int const dw = dst_rect.width();
    int const dh = dst_rect.height();
    WeightCalculator calc_weights;

    #pragma omp parallel for schedule(static) shared(inv_xform)
    for (int dy = 0; dy < dh; ++dy) {
        uint32_t* dst_line = dst_data + dy * dst_stride;
        double const f_dy_center = dy + 0.5;
        double const f_sx_base = f_dy_center * inv_xform.m21() + inv_xform.dx();
        double const f_sy_base = f_dy_center * inv_xform.m22() + inv_xform.dy();

        for (int dx = 0; dx < dw; ++dx) {
            double const f_dx_center = dx + 0.5;
            double const sx = f_sx_base + f_dx_center * inv_xform.m11();
            double const sy = f_sy_base + f_dx_center * inv_xform.m12();

            double const x_prime = sx - 0.5;
            double const y_prime = sy - 0.5;

            int const x0 = (int)std::floor(x_prime);
            int const y0 = (int)std::floor(y_prime);

            if (x0 < -2 || x0 >= sw + 1 || y0 < -2 || y0 >= sh + 1) {
                if (outside_flags & OutsidePixels::COLOR) {
                    dst_line[dx] = outside_color;
                } else {
                    int const cx = qBound(0, x0, sw - 1);
                    int const cy = qBound(0, y0, sh - 1);
                    dst_line[dx] = src_data[cy * src_stride + cx];
                }
                continue;
            }

            float const u = (float)(x_prime - x0);
            float const v = (float)(y_prime - y0);

            float wx_m1, wx_0, wx_1, wx_2;
            calc_weights(u, wx_m1, wx_0, wx_1, wx_2);

            float wy_m1, wy_0, wy_1, wy_2;
            calc_weights(v, wy_m1, wy_0, wy_1, wy_2);

            float r_final, g_final, b_final;

            if (x0 >= 1 && x0 + 2 < sw && y0 >= 1 && y0 + 2 < sh) {
                uint32_t const* p_row_m1 = src_data + (y0 - 1) * src_stride + (x0 - 1);
                uint32_t const* p_row_0  = p_row_m1 + src_stride;
                uint32_t const* p_row_1  = p_row_0 + src_stride;
                uint32_t const* p_row_2  = p_row_1 + src_stride;

                auto row_interp = [&](uint32_t const* p, float& r, float& g, float& b) {
                    uint32_t const c0 = p[0], c1 = p[1], c2 = p[2], c3 = p[3];
                    r = wx_m1 * ((c0 >> 16) & 0xFF) + wx_0 * ((c1 >> 16) & 0xFF) + wx_1 * ((c2 >> 16) & 0xFF) + wx_2 * ((c3 >> 16) & 0xFF);
                    g = wx_m1 * ((c0 >> 8) & 0xFF)  + wx_0 * ((c1 >> 8) & 0xFF)  + wx_1 * ((c2 >> 8) & 0xFF)  + wx_2 * ((c3 >> 8) & 0xFF);
                    b = wx_m1 * (c0 & 0xFF)         + wx_0 * (c1 & 0xFF)         + wx_1 * (c2 & 0xFF)         + wx_2 * (c3 & 0xFF);
                };

                float r_m1, g_m1, b_m1, r_0, g_0, b_0, r_1, g_1, b_1, r_2, g_2, b_2;
                row_interp(p_row_m1, r_m1, g_m1, b_m1);
                row_interp(p_row_0,  r_0,  g_0,  b_0);
                row_interp(p_row_1,  r_1,  g_1,  b_1);
                row_interp(p_row_2,  r_2,  g_2,  b_2);

                r_final = wy_m1 * r_m1 + wy_0 * r_0 + wy_1 * r_1 + wy_2 * r_2;
                g_final = wy_m1 * g_m1 + wy_0 * g_0 + wy_1 * g_1 + wy_2 * g_2;
                b_final = wy_m1 * b_m1 + wy_0 * b_0 + wy_1 * b_1 + wy_2 * b_2;
            } else {
                auto fetch = [&](int const x, int const y) -> uint32_t {
                    if (x >= 0 && x < sw && y >= 0 && y < sh) {
                        return src_data[y * src_stride + x];
                    }
                    if (outside_flags & OutsidePixels::COLOR) {
                        return outside_color;
                    }
                    int const cx = qBound(0, x, sw - 1);
                    int const cy = qBound(0, y, sh - 1);
                    return src_data[cy * src_stride + cx];
                };

                auto row_interp_safe = [&](int const y, float& r, float& g, float& b) {
                    uint32_t const c0 = fetch(x0 - 1, y), c1 = fetch(x0, y), c2 = fetch(x0 + 1, y), c3 = fetch(x0 + 2, y);
                    r = wx_m1 * ((c0 >> 16) & 0xFF) + wx_0 * ((c1 >> 16) & 0xFF) + wx_1 * ((c2 >> 16) & 0xFF) + wx_2 * ((c3 >> 16) & 0xFF);
                    g = wx_m1 * ((c0 >> 8) & 0xFF)  + wx_0 * ((c1 >> 8) & 0xFF)  + wx_1 * ((c2 >> 8) & 0xFF)  + wx_2 * ((c3 >> 8) & 0xFF);
                    b = wx_m1 * (c0 & 0xFF)         + wx_0 * (c1 & 0xFF)         + wx_1 * (c2 & 0xFF)         + wx_2 * (c3 & 0xFF);
                };

                float r_m1, g_m1, b_m1, r_0, g_0, b_0, r_1, g_1, b_1, r_2, g_2, b_2;
                row_interp_safe(y0 - 1, r_m1, g_m1, b_m1);
                row_interp_safe(y0,     r_0,  g_0,  b_0);
                row_interp_safe(y0 + 1, r_1,  g_1,  b_1);
                row_interp_safe(y0 + 2, r_2,  g_2,  b_2);

                r_final = wy_m1 * r_m1 + wy_0 * r_0 + wy_1 * r_1 + wy_2 * r_2;
                g_final = wy_m1 * g_m1 + wy_0 * g_0 + wy_1 * g_1 + wy_2 * g_2;
                b_final = wy_m1 * b_m1 + wy_0 * b_0 + wy_1 * b_1 + wy_2 * b_2;
            }

            int const ir = qBound(0, (int)std::round(r_final), 255);
            int const ig = qBound(0, (int)std::round(g_final), 255);
            int const ib = qBound(0, (int)std::round(b_final), 255);
            dst_line[dx] = 0xFF000000 | (ir << 16) | (ig << 8) | ib;
        }
    }
}

template<typename WeightCalculator>
static void transformBicubicARGB32(
    uint32_t const* const src_data, int const src_stride, QSize const src_size,
    uint32_t* const dst_data, int const dst_stride,
    QTransform const& inv_xform, QRect const& dst_rect,
    uint32_t const outside_color, int const outside_flags)
{
    int const sw = src_size.width();
    int const sh = src_size.height();
    int const dw = dst_rect.width();
    int const dh = dst_rect.height();
    WeightCalculator calc_weights;

    #pragma omp parallel for schedule(static) shared(inv_xform)
    for (int dy = 0; dy < dh; ++dy) {
        uint32_t* dst_line = dst_data + dy * dst_stride;
        double const f_dy_center = dy + 0.5;
        double const f_sx_base = f_dy_center * inv_xform.m21() + inv_xform.dx();
        double const f_sy_base = f_dy_center * inv_xform.m22() + inv_xform.dy();

        for (int dx = 0; dx < dw; ++dx) {
            double const f_dx_center = dx + 0.5;
            double const sx = f_sx_base + f_dx_center * inv_xform.m11();
            double const sy = f_sy_base + f_dx_center * inv_xform.m12();

            double const x_prime = sx - 0.5;
            double const y_prime = sy - 0.5;

            int const x0 = (int)std::floor(x_prime);
            int const y0 = (int)std::floor(y_prime);

            if (x0 < -2 || x0 >= sw + 1 || y0 < -2 || y0 >= sh + 1) {
                if (outside_flags & OutsidePixels::COLOR) {
                    dst_line[dx] = outside_color;
                } else {
                    int const cx = qBound(0, x0, sw - 1);
                    int const cy = qBound(0, y0, sh - 1);
                    dst_line[dx] = src_data[cy * src_stride + cx];
                }
                continue;
            }

            float const u = (float)(x_prime - x0);
            float const v = (float)(y_prime - y0);

            float wx_m1, wx_0, wx_1, wx_2;
            calc_weights(u, wx_m1, wx_0, wx_1, wx_2);

            float wy_m1, wy_0, wy_1, wy_2;
            calc_weights(v, wy_m1, wy_0, wy_1, wy_2);

            float a_final, r_final, g_final, b_final;

            if (x0 >= 1 && x0 + 2 < sw && y0 >= 1 && y0 + 2 < sh) {
                uint32_t const* p_row_m1 = src_data + (y0 - 1) * src_stride + (x0 - 1);
                uint32_t const* p_row_0  = p_row_m1 + src_stride;
                uint32_t const* p_row_1  = p_row_0 + src_stride;
                uint32_t const* p_row_2  = p_row_1 + src_stride;

                auto row_interp = [&](uint32_t const* p, float& a, float& r, float& g, float& b) {
                    uint32_t const c0 = p[0], c1 = p[1], c2 = p[2], c3 = p[3];
                    a = wx_m1 * ((c0 >> 24) & 0xFF) + wx_0 * ((c1 >> 24) & 0xFF) + wx_1 * ((c2 >> 24) & 0xFF) + wx_2 * ((c3 >> 24) & 0xFF);
                    r = wx_m1 * ((c0 >> 16) & 0xFF) + wx_0 * ((c1 >> 16) & 0xFF) + wx_1 * ((c2 >> 16) & 0xFF) + wx_2 * ((c3 >> 16) & 0xFF);
                    g = wx_m1 * ((c0 >> 8) & 0xFF)  + wx_0 * ((c1 >> 8) & 0xFF)  + wx_1 * ((c2 >> 8) & 0xFF)  + wx_2 * ((c3 >> 8) & 0xFF);
                    b = wx_m1 * (c0 & 0xFF)         + wx_0 * (c1 & 0xFF)         + wx_1 * (c2 & 0xFF)         + wx_2 * (c3 & 0xFF);
                };

                float a_m1, r_m1, g_m1, b_m1, a_0, r_0, g_0, b_0, a_1, r_1, g_1, b_1, a_2, r_2, g_2, b_2;
                row_interp(p_row_m1, a_m1, r_m1, g_m1, b_m1);
                row_interp(p_row_0,  a_0,  r_0,  g_0,  b_0);
                row_interp(p_row_1,  a_1,  r_1,  g_1,  b_1);
                row_interp(p_row_2,  a_2,  r_2,  g_2,  b_2);

                a_final = wy_m1 * a_m1 + wy_0 * a_0 + wy_1 * a_1 + wy_2 * a_2;
                r_final = wy_m1 * r_m1 + wy_0 * r_0 + wy_1 * r_1 + wy_2 * r_2;
                g_final = wy_m1 * g_m1 + wy_0 * g_0 + wy_1 * g_1 + wy_2 * g_2;
                b_final = wy_m1 * b_m1 + wy_0 * b_0 + wy_1 * b_1 + wy_2 * b_2;
            } else {
                auto fetch = [&](int const x, int const y) -> uint32_t {
                    if (x >= 0 && x < sw && y >= 0 && y < sh) {
                        return src_data[y * src_stride + x];
                    }
                    if (outside_flags & OutsidePixels::COLOR) {
                        return outside_color;
                    }
                    int const cx = qBound(0, x, sw - 1);
                    int const cy = qBound(0, y, sh - 1);
                    return src_data[cy * src_stride + cx];
                };

                auto row_interp_safe = [&](int const y, float& a, float& r, float& g, float& b) {
                    uint32_t const c0 = fetch(x0 - 1, y), c1 = fetch(x0, y), c2 = fetch(x0 + 1, y), c3 = fetch(x0 + 2, y);
                    a = wx_m1 * ((c0 >> 24) & 0xFF) + wx_0 * ((c1 >> 24) & 0xFF) + wx_1 * ((c2 >> 24) & 0xFF) + wx_2 * ((c3 >> 24) & 0xFF);
                    r = wx_m1 * ((c0 >> 16) & 0xFF) + wx_0 * ((c1 >> 16) & 0xFF) + wx_1 * ((c2 >> 16) & 0xFF) + wx_2 * ((c3 >> 16) & 0xFF);
                    g = wx_m1 * ((c0 >> 8) & 0xFF)  + wx_0 * ((c1 >> 8) & 0xFF)  + wx_1 * ((c2 >> 8) & 0xFF)  + wx_2 * ((c3 >> 8) & 0xFF);
                    b = wx_m1 * (c0 & 0xFF)         + wx_0 * (c1 & 0xFF)         + wx_1 * (c2 & 0xFF)         + wx_2 * (c3 & 0xFF);
                };

                float a_m1, r_m1, g_m1, b_m1, a_0, r_0, g_0, b_0, a_1, r_1, g_1, b_1, a_2, r_2, g_2, b_2;
                row_interp_safe(y0 - 1, a_m1, r_m1, g_m1, b_m1);
                row_interp_safe(y0,     a_0,  r_0,  g_0,  b_0);
                row_interp_safe(y0 + 1, a_1,  r_1,  g_1,  b_1);
                row_interp_safe(y0 + 2, a_2,  r_2,  g_2,  b_2);

                a_final = wy_m1 * a_m1 + wy_0 * a_0 + wy_1 * a_1 + wy_2 * a_2;
                r_final = wy_m1 * r_m1 + wy_0 * r_0 + wy_1 * r_1 + wy_2 * r_2;
                g_final = wy_m1 * g_m1 + wy_0 * g_0 + wy_1 * g_1 + wy_2 * g_2;
                b_final = wy_m1 * b_m1 + wy_0 * b_0 + wy_1 * b_1 + wy_2 * b_2;
            }

            int const ia = qBound(0, (int)std::round(a_final), 255);
            int const ir = qBound(0, (int)std::round(r_final), 255);
            int const ig = qBound(0, (int)std::round(g_final), 255);
            int const ib = qBound(0, (int)std::round(b_final), 255);
            dst_line[dx] = (ia << 24) | (ir << 16) | (ig << 8) | ib;
        }
    }
}

static void transformBilinearGray(
    uint8_t const* const src_data, int const src_stride, QSize const src_size,
    uint8_t* const dst_data, int const dst_stride,
    QTransform const& inv_xform, QRect const& dst_rect,
    uint8_t const outside_color, int const outside_flags)
{
    int const sw = src_size.width();
    int const sh = src_size.height();
    int const dw = dst_rect.width();
    int const dh = dst_rect.height();

    #pragma omp parallel for schedule(static) shared(inv_xform)
    for (int dy = 0; dy < dh; ++dy) {
        uint8_t* dst_line = dst_data + dy * dst_stride;
        double const f_dy_center = dy + 0.5;
        double const f_sx_base = f_dy_center * inv_xform.m21() + inv_xform.dx();
        double const f_sy_base = f_dy_center * inv_xform.m22() + inv_xform.dy();

        for (int dx = 0; dx < dw; ++dx) {
            double const f_dx_center = dx + 0.5;
            double const sx = f_sx_base + f_dx_center * inv_xform.m11();
            double const sy = f_sy_base + f_dx_center * inv_xform.m12();

            double const x_prime = sx - 0.5;
            double const y_prime = sy - 0.5;

            int const x0 = (int)std::floor(x_prime);
            int const y0 = (int)std::floor(y_prime);

            if (x0 < -1 || x0 >= sw || y0 < -1 || y0 >= sh) {
                if (outside_flags & OutsidePixels::COLOR) {
                    dst_line[dx] = outside_color;
                } else {
                    int const cx = qBound(0, x0, sw - 1);
                    int const cy = qBound(0, y0, sh - 1);
                    dst_line[dx] = src_data[cy * src_stride + cx];
                }
                continue;
            }

            float const u = (float)(x_prime - x0);
            float const v = (float)(y_prime - y0);

            float const wx_0 = 1.0f - u, wx_1 = u;
            float const wy_0 = 1.0f - v, wy_1 = v;

            if (x0 >= 0 && x0 + 1 < sw && y0 >= 0 && y0 + 1 < sh) {
                uint8_t const* p_row_0 = src_data + y0 * src_stride + x0;
                uint8_t const* p_row_1 = p_row_0 + src_stride;
                float const r_0 = wx_0 * p_row_0[0] + wx_1 * p_row_0[1];
                float const r_1 = wx_0 * p_row_1[0] + wx_1 * p_row_1[1];
                float const val = wy_0 * r_0 + wy_1 * r_1;
                int const ival = (int)std::round(val);
                dst_line[dx] = (uint8_t)qBound(0, ival, 255);
            } else {
                auto fetch = [&](int const x, int const y) -> float {
                    if (x >= 0 && x < sw && y >= 0 && y < sh) {
                        return (float)src_data[y * src_stride + x];
                    }
                    if (outside_flags & OutsidePixels::COLOR) {
                        return (float)outside_color;
                    }
                    int const cx = qBound(0, x, sw - 1);
                    int const cy = qBound(0, y, sh - 1);
                    return (float)src_data[cy * src_stride + cx];
                };
                float const r_0 = wx_0 * fetch(x0, y0)     + wx_1 * fetch(x0 + 1, y0);
                float const r_1 = wx_0 * fetch(x0, y0 + 1) + wx_1 * fetch(x0 + 1, y0 + 1);
                float const val = wy_0 * r_0 + wy_1 * r_1;
                int const ival = (int)std::round(val);
                dst_line[dx] = (uint8_t)qBound(0, ival, 255);
            }
        }
    }
}

static void transformBilinearRGB32(
    uint32_t const* const src_data, int const src_stride, QSize const src_size,
    uint32_t* const dst_data, int const dst_stride,
    QTransform const& inv_xform, QRect const& dst_rect,
    uint32_t const outside_color, int const outside_flags)
{
    int const sw = src_size.width();
    int const sh = src_size.height();
    int const dw = dst_rect.width();
    int const dh = dst_rect.height();

    #pragma omp parallel for schedule(static) shared(inv_xform)
    for (int dy = 0; dy < dh; ++dy) {
        uint32_t* dst_line = dst_data + dy * dst_stride;
        double const f_dy_center = dy + 0.5;
        double const f_sx_base = f_dy_center * inv_xform.m21() + inv_xform.dx();
        double const f_sy_base = f_dy_center * inv_xform.m22() + inv_xform.dy();

        for (int dx = 0; dx < dw; ++dx) {
            double const f_dx_center = dx + 0.5;
            double const sx = f_sx_base + f_dx_center * inv_xform.m11();
            double const sy = f_sy_base + f_dx_center * inv_xform.m12();

            double const x_prime = sx - 0.5;
            double const y_prime = sy - 0.5;

            int const x0 = (int)std::floor(x_prime);
            int const y0 = (int)std::floor(y_prime);

            if (x0 < -1 || x0 >= sw || y0 < -1 || y0 >= sh) {
                if (outside_flags & OutsidePixels::COLOR) {
                    dst_line[dx] = outside_color;
                } else {
                    int const cx = qBound(0, x0, sw - 1);
                    int const cy = qBound(0, y0, sh - 1);
                    dst_line[dx] = src_data[cy * src_stride + cx];
                }
                continue;
            }

            float const u = (float)(x_prime - x0);
            float const v = (float)(y_prime - y0);

            float const wx_0 = 1.0f - u, wx_1 = u;
            float const wy_0 = 1.0f - v, wy_1 = v;

            float r_final, g_final, b_final;

            if (x0 >= 0 && x0 + 1 < sw && y0 >= 0 && y0 + 1 < sh) {
                uint32_t const* p_row_0 = src_data + y0 * src_stride + x0;
                uint32_t const* p_row_1 = p_row_0 + src_stride;

                uint32_t const c00 = p_row_0[0], c01 = p_row_0[1];
                uint32_t const c10 = p_row_1[0], c11 = p_row_1[1];

                float const r0 = wx_0 * ((c00 >> 16) & 0xFF) + wx_1 * ((c01 >> 16) & 0xFF);
                float const g0 = wx_0 * ((c00 >> 8) & 0xFF)  + wx_1 * ((c01 >> 8) & 0xFF);
                float const b0 = wx_0 * (c00 & 0xFF)         + wx_1 * (c01 & 0xFF);

                float const r1 = wx_0 * ((c10 >> 16) & 0xFF) + wx_1 * ((c11 >> 16) & 0xFF);
                float const g1 = wx_0 * ((c10 >> 8) & 0xFF)  + wx_1 * ((c11 >> 8) & 0xFF);
                float const b1 = wx_0 * (c10 & 0xFF)         + wx_1 * (c11 & 0xFF);

                r_final = wy_0 * r0 + wy_1 * r1;
                g_final = wy_0 * g0 + wy_1 * g1;
                b_final = wy_0 * b0 + wy_1 * b1;
            } else {
                auto fetch = [&](int const x, int const y) -> uint32_t {
                    if (x >= 0 && x < sw && y >= 0 && y < sh) {
                        return src_data[y * src_stride + x];
                    }
                    if (outside_flags & OutsidePixels::COLOR) {
                        return outside_color;
                    }
                    int const cx = qBound(0, x, sw - 1);
                    int const cy = qBound(0, y, sh - 1);
                    return src_data[cy * src_stride + cx];
                };

                uint32_t const c00 = fetch(x0, y0),     c01 = fetch(x0 + 1, y0);
                uint32_t const c10 = fetch(x0, y0 + 1), c11 = fetch(x0 + 1, y0 + 1);

                float const r0 = wx_0 * ((c00 >> 16) & 0xFF) + wx_1 * ((c01 >> 16) & 0xFF);
                float const g0 = wx_0 * ((c00 >> 8) & 0xFF)  + wx_1 * ((c01 >> 8) & 0xFF);
                float const b0 = wx_0 * (c00 & 0xFF)         + wx_1 * (c01 & 0xFF);

                float const r1 = wx_0 * ((c10 >> 16) & 0xFF) + wx_1 * ((c11 >> 16) & 0xFF);
                float const g1 = wx_0 * ((c10 >> 8) & 0xFF)  + wx_1 * ((c11 >> 8) & 0xFF);
                float const b1 = wx_0 * (c10 & 0xFF)         + wx_1 * (c11 & 0xFF);

                r_final = wy_0 * r0 + wy_1 * r1;
                g_final = wy_0 * g0 + wy_1 * g1;
                b_final = wy_0 * b0 + wy_1 * b1;
            }

            int const ir = qBound(0, (int)std::round(r_final), 255);
            int const ig = qBound(0, (int)std::round(g_final), 255);
            int const ib = qBound(0, (int)std::round(b_final), 255);
            dst_line[dx] = 0xFF000000 | (ir << 16) | (ig << 8) | ib;
        }
    }
}

static void transformBilinearARGB32(
    uint32_t const* const src_data, int const src_stride, QSize const src_size,
    uint32_t* const dst_data, int const dst_stride,
    QTransform const& inv_xform, QRect const& dst_rect,
    uint32_t const outside_color, int const outside_flags)
{
    int const sw = src_size.width();
    int const sh = src_size.height();
    int const dw = dst_rect.width();
    int const dh = dst_rect.height();

    #pragma omp parallel for schedule(static) shared(inv_xform)
    for (int dy = 0; dy < dh; ++dy) {
        uint32_t* dst_line = dst_data + dy * dst_stride;
        double const f_dy_center = dy + 0.5;
        double const f_sx_base = f_dy_center * inv_xform.m21() + inv_xform.dx();
        double const f_sy_base = f_dy_center * inv_xform.m22() + inv_xform.dy();

        for (int dx = 0; dx < dw; ++dx) {
            double const f_dx_center = dx + 0.5;
            double const sx = f_sx_base + f_dx_center * inv_xform.m11();
            double const sy = f_sy_base + f_dx_center * inv_xform.m12();

            double const x_prime = sx - 0.5;
            double const y_prime = sy - 0.5;

            int const x0 = (int)std::floor(x_prime);
            int const y0 = (int)std::floor(y_prime);

            if (x0 < -1 || x0 >= sw || y0 < -1 || y0 >= sh) {
                if (outside_flags & OutsidePixels::COLOR) {
                    dst_line[dx] = outside_color;
                } else {
                    int const cx = qBound(0, x0, sw - 1);
                    int const cy = qBound(0, y0, sh - 1);
                    dst_line[dx] = src_data[cy * src_stride + cx];
                }
                continue;
            }

            float const u = (float)(x_prime - x0);
            float const v = (float)(y_prime - y0);

            float const wx_0 = 1.0f - u, wx_1 = u;
            float const wy_0 = 1.0f - v, wy_1 = v;

            float a_final, r_final, g_final, b_final;

            if (x0 >= 0 && x0 + 1 < sw && y0 >= 0 && y0 + 1 < sh) {
                uint32_t const* p_row_0 = src_data + y0 * src_stride + x0;
                uint32_t const* p_row_1 = p_row_0 + src_stride;

                uint32_t const c00 = p_row_0[0], c01 = p_row_0[1];
                uint32_t const c10 = p_row_1[0], c11 = p_row_1[1];

                float const a0 = wx_0 * ((c00 >> 24) & 0xFF) + wx_1 * ((c01 >> 24) & 0xFF);
                float const r0 = wx_0 * ((c00 >> 16) & 0xFF) + wx_1 * ((c01 >> 16) & 0xFF);
                float const g0 = wx_0 * ((c00 >> 8) & 0xFF)  + wx_1 * ((c01 >> 8) & 0xFF);
                float const b0 = wx_0 * (c00 & 0xFF)         + wx_1 * (c01 & 0xFF);

                float const a1 = wx_0 * ((c10 >> 24) & 0xFF) + wx_1 * ((c11 >> 24) & 0xFF);
                float const r1 = wx_0 * ((c10 >> 16) & 0xFF) + wx_1 * ((c11 >> 16) & 0xFF);
                float const g1 = wx_0 * ((c10 >> 8) & 0xFF)  + wx_1 * ((c11 >> 8) & 0xFF);
                float const b1 = wx_0 * (c10 & 0xFF)         + wx_1 * (c11 & 0xFF);

                a_final = wy_0 * a0 + wy_1 * a1;
                r_final = wy_0 * r0 + wy_1 * r1;
                g_final = wy_0 * g0 + wy_1 * g1;
                b_final = wy_0 * b0 + wy_1 * b1;
            } else {
                auto fetch = [&](int const x, int const y) -> uint32_t {
                    if (x >= 0 && x < sw && y >= 0 && y < sh) {
                        return src_data[y * src_stride + x];
                    }
                    if (outside_flags & OutsidePixels::COLOR) {
                        return outside_color;
                    }
                    int const cx = qBound(0, x, sw - 1);
                    int const cy = qBound(0, y, sh - 1);
                    return src_data[cy * src_stride + cx];
                };

                uint32_t const c00 = fetch(x0, y0),     c01 = fetch(x0 + 1, y0);
                uint32_t const c10 = fetch(x0, y0 + 1), c11 = fetch(x0 + 1, y0 + 1);

                float const a0 = wx_0 * ((c00 >> 24) & 0xFF) + wx_1 * ((c01 >> 24) & 0xFF);
                float const r0 = wx_0 * ((c00 >> 16) & 0xFF) + wx_1 * ((c01 >> 16) & 0xFF);
                float const g0 = wx_0 * ((c00 >> 8) & 0xFF)  + wx_1 * ((c01 >> 8) & 0xFF);
                float const b0 = wx_0 * (c00 & 0xFF)         + wx_1 * (c01 & 0xFF);

                float const a1 = wx_0 * ((c10 >> 24) & 0xFF) + wx_1 * ((c11 >> 24) & 0xFF);
                float const r1 = wx_0 * ((c10 >> 16) & 0xFF) + wx_1 * ((c11 >> 16) & 0xFF);
                float const g1 = wx_0 * ((c10 >> 8) & 0xFF)  + wx_1 * ((c11 >> 8) & 0xFF);
                float const b1 = wx_0 * (c10 & 0xFF)         + wx_1 * (c11 & 0xFF);

                a_final = wy_0 * a0 + wy_1 * a1;
                r_final = wy_0 * r0 + wy_1 * r1;
                g_final = wy_0 * g0 + wy_1 * g1;
                b_final = wy_0 * b0 + wy_1 * b1;
            }

            int const ia = qBound(0, (int)std::round(a_final), 255);
            int const ir = qBound(0, (int)std::round(r_final), 255);
            int const ig = qBound(0, (int)std::round(g_final), 255);
            int const ib = qBound(0, (int)std::round(b_final), 255);
            dst_line[dx] = (ia << 24) | (ir << 16) | (ig << 8) | ib;
        }
    }
}

static bool isUpscalingTransform(QTransform const& inv_xform)
{
    QPolygonF dst_poly;
    dst_poly.push_back(QPointF(0.5, 0.0));
    dst_poly.push_back(QPointF(1.0, 0.5));
    dst_poly.push_back(QPointF(0.5, 1.0));
    dst_poly.push_back(QPointF(0.0, 0.5));

    QPolygonF src_poly(inv_xform.map(dst_poly));
    std::sort(src_poly.begin(), src_poly.end(), XLess());
    double const width = src_poly.back().x() - src_poly.front().x();
    std::sort(src_poly.begin(), src_poly.end(), YLess());
    double const height = src_poly.back().y() - src_poly.front().y();

    return (width < 0.999 || height < 0.999);
}

} // anonymous namespace

QImage transform(
    QImage const& src, QTransform const& xform,
    QRect const& dst_rect, OutsidePixels const outside_pixels,
    QSizeF const& min_mapping_area,
    UpscalingMethod upscaling_method)
{
    if (src.isNull() || dst_rect.isEmpty()) {
        return QImage();
    }

    if (!xform.isAffine()) {
        throw std::invalid_argument("transform: only affine transformations are supported");
    }

    if (!dst_rect.isValid()) {
        throw std::invalid_argument("transform: dst_rect is invalid");
    }

    if (upscaling_method == UPSCALING_AUTO) {
        upscaling_method = defaultUpscalingMethod();
    }

    QTransform inv_xform;
    inv_xform.translate(dst_rect.x(), dst_rect.y());
    inv_xform *= xform.inverted();

    bool const upscaling = (upscaling_method != UPSCALING_BOX) && isUpscalingTransform(inv_xform);

    if (src.format() == QImage::Format_Indexed8 && src.allGray()) {
        GrayImage gray_src(src);
        GrayImage gray_dst(dst_rect.size());

        if (upscaling) {
            if (upscaling_method == UPSCALING_BICUBIC_MITCHELL) {
                transformBicubicGray<MitchellWeights>(
                    gray_src.data(), gray_src.stride(), src.size(),
                    gray_dst.data(), gray_dst.stride(), inv_xform, dst_rect,
                    outside_pixels.grayLevel(), outside_pixels.flags()
                );
            } else if (upscaling_method == UPSCALING_BILINEAR) {
                transformBilinearGray(
                    gray_src.data(), gray_src.stride(), src.size(),
                    gray_dst.data(), gray_dst.stride(), inv_xform, dst_rect,
                    outside_pixels.grayLevel(), outside_pixels.flags()
                );
            } else {
                transformBicubicGray<CatmullRomWeights>(
                    gray_src.data(), gray_src.stride(), src.size(),
                    gray_dst.data(), gray_dst.stride(), inv_xform, dst_rect,
                    outside_pixels.grayLevel(), outside_pixels.flags()
                );
            }
        } else {
            transformGeneric<uint8_t, Gray>(
                gray_src.data(), gray_src.stride(), src.size(),
                gray_dst.data(), gray_dst.stride(), xform, dst_rect,
                outside_pixels.grayLevel(), outside_pixels.flags(),
                min_mapping_area
            );
        }
        return gray_dst;
    } else {
        if (src.hasAlphaChannel() || qAlpha(outside_pixels.rgba()) != 0xff) {
            QImage const src_argb32(src.convertToFormat(QImage::Format_ARGB32));
            QImage dst(dst_rect.size(), QImage::Format_ARGB32);

            if (upscaling) {
                if (upscaling_method == UPSCALING_BICUBIC_MITCHELL) {
                    transformBicubicARGB32<MitchellWeights>(
                        (uint32_t const*)src_argb32.bits(), src_argb32.bytesPerLine() / 4, src_argb32.size(),
                        (uint32_t*)dst.bits(), dst.bytesPerLine() / 4, inv_xform, dst_rect,
                        outside_pixels.rgba(), outside_pixels.flags()
                    );
                } else if (upscaling_method == UPSCALING_BILINEAR) {
                    transformBilinearARGB32(
                        (uint32_t const*)src_argb32.bits(), src_argb32.bytesPerLine() / 4, src_argb32.size(),
                        (uint32_t*)dst.bits(), dst.bytesPerLine() / 4, inv_xform, dst_rect,
                        outside_pixels.rgba(), outside_pixels.flags()
                    );
                } else {
                    transformBicubicARGB32<CatmullRomWeights>(
                        (uint32_t const*)src_argb32.bits(), src_argb32.bytesPerLine() / 4, src_argb32.size(),
                        (uint32_t*)dst.bits(), dst.bytesPerLine() / 4, inv_xform, dst_rect,
                        outside_pixels.rgba(), outside_pixels.flags()
                    );
                }
            } else {
                transformGeneric<uint32_t, ARGB32>(
                    (uint32_t const*)src_argb32.bits(), src_argb32.bytesPerLine() / 4, src_argb32.size(),
                    (uint32_t*)dst.bits(), dst.bytesPerLine() / 4, xform, dst_rect,
                    outside_pixels.rgba(), outside_pixels.flags(), min_mapping_area
                );
            }
            return dst;
        } else {
            QImage const src_rgb32(src.convertToFormat(QImage::Format_RGB32));
            QImage dst(dst_rect.size(), QImage::Format_RGB32);

            if (upscaling) {
                if (upscaling_method == UPSCALING_BICUBIC_MITCHELL) {
                    transformBicubicRGB32<MitchellWeights>(
                        (uint32_t const*)src_rgb32.bits(), src_rgb32.bytesPerLine() / 4, src_rgb32.size(),
                        (uint32_t*)dst.bits(), dst.bytesPerLine() / 4, inv_xform, dst_rect,
                        outside_pixels.rgb(), outside_pixels.flags()
                    );
                } else if (upscaling_method == UPSCALING_BILINEAR) {
                    transformBilinearRGB32(
                        (uint32_t const*)src_rgb32.bits(), src_rgb32.bytesPerLine() / 4, src_rgb32.size(),
                        (uint32_t*)dst.bits(), dst.bytesPerLine() / 4, inv_xform, dst_rect,
                        outside_pixels.rgb(), outside_pixels.flags()
                    );
                } else {
                    transformBicubicRGB32<CatmullRomWeights>(
                        (uint32_t const*)src_rgb32.bits(), src_rgb32.bytesPerLine() / 4, src_rgb32.size(),
                        (uint32_t*)dst.bits(), dst.bytesPerLine() / 4, inv_xform, dst_rect,
                        outside_pixels.rgb(), outside_pixels.flags()
                    );
                }
            } else {
                transformGeneric<uint32_t, RGB32>(
                    (uint32_t const*)src_rgb32.bits(), src_rgb32.bytesPerLine() / 4, src_rgb32.size(),
                    (uint32_t*)dst.bits(), dst.bytesPerLine() / 4, xform, dst_rect,
                    outside_pixels.rgb(), outside_pixels.flags(), min_mapping_area
                );
            }
            return dst;
        }
    }
}

GrayImage transformToGray(
    QImage const& src, QTransform const& xform,
    QRect const& dst_rect, OutsidePixels const outside_pixels,
    QSizeF const& min_mapping_area,
    UpscalingMethod upscaling_method)
{
    if (src.isNull() || dst_rect.isEmpty()) {
        return GrayImage();
    }

    if (!xform.isAffine()) {
        throw std::invalid_argument("transformToGray: only affine transformations are supported");
    }

    if (!dst_rect.isValid()) {
        throw std::invalid_argument("transformToGray: dst_rect is invalid");
    }

    if (upscaling_method == UPSCALING_AUTO) {
        upscaling_method = defaultUpscalingMethod();
    }

    QTransform inv_xform;
    inv_xform.translate(dst_rect.x(), dst_rect.y());
    inv_xform *= xform.inverted();

    bool const upscaling = (upscaling_method != UPSCALING_BOX) && isUpscalingTransform(inv_xform);

    GrayImage const gray_src(src);
    GrayImage dst(dst_rect.size());

    if (upscaling) {
        if (upscaling_method == UPSCALING_BICUBIC_MITCHELL) {
            transformBicubicGray<MitchellWeights>(
                gray_src.data(), gray_src.stride(), gray_src.size(),
                dst.data(), dst.stride(), inv_xform, dst_rect,
                outside_pixels.grayLevel(), outside_pixels.flags()
            );
        } else if (upscaling_method == UPSCALING_BILINEAR) {
            transformBilinearGray(
                gray_src.data(), gray_src.stride(), gray_src.size(),
                dst.data(), dst.stride(), inv_xform, dst_rect,
                outside_pixels.grayLevel(), outside_pixels.flags()
            );
        } else {
            transformBicubicGray<CatmullRomWeights>(
                gray_src.data(), gray_src.stride(), gray_src.size(),
                dst.data(), dst.stride(), inv_xform, dst_rect,
                outside_pixels.grayLevel(), outside_pixels.flags()
            );
        }
    } else {
        transformGeneric<uint8_t, Gray>(
            gray_src.data(), gray_src.stride(), gray_src.size(),
            dst.data(), dst.stride(), xform, dst_rect,
            outside_pixels.grayLevel(), outside_pixels.flags(),
            min_mapping_area
        );
    }

    return dst;
}

} // namespace imageproc
