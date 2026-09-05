/*
    Scan Tailor - Interactive post-processing tool for scanned pages.
    Copyright (C) 2007-2008  Joseph Artsimovich <joseph_a@mail.ru>

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
#include "Utils.h"
#include "UpscalingMethod.h"
#include <QImage>
#include <QSize>
#include <QTransform>
#ifndef Q_MOC_RUN
#include <boost/test/unit_test.hpp>
#endif
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

namespace imageproc
{

namespace tests
{

using namespace utils;

BOOST_AUTO_TEST_SUITE(TransformTestSuite);

BOOST_AUTO_TEST_CASE(test_null_image)
{
    QImage const null_img;
    QTransform const null_xform;
    QRect const unit_rect(0, 0, 1, 1);
    QColor const bgcolor(0xff, 0xff, 0xff);
    OutsidePixels const outside_pixels(OutsidePixels::assumeColor(bgcolor));
    BOOST_CHECK(transformToGray(null_img, null_xform, unit_rect, outside_pixels).isNull());
}

BOOST_AUTO_TEST_CASE(test_random_image)
{
    GrayImage img(QSize(100, 100));
    uint8_t* line = img.data();
    for (int y = 0; y < img.height(); ++y) {
        for (int x = 0; x < img.width(); ++x) {
            line[x] = rand() % 256;
        }
        line += img.stride();
    }

    QColor const bgcolor(0xff, 0xff, 0xff);
    OutsidePixels const outside_pixels(OutsidePixels::assumeColor(bgcolor));

    QTransform const null_xform;
    BOOST_CHECK(transformToGray(img, null_xform, img.rect(), outside_pixels) == img);
}

BOOST_AUTO_TEST_CASE(test_upscaling_bicubic_smooth_gradient)
{
    // A 2x1 image with a hard transition between black (0) and white (255).
    int const src_data[] = {0, 255};
    QImage const src = utils::makeGrayImage(src_data, 2, 1);

    // Upscale 2x horizontally: 2x1 -> 4x1.
    // The transform maps source coordinates to destination coordinates.
    QTransform const xform = QTransform().scale(2.0, 1.0);
    QRect const dst_rect(0, 0, 4, 1);
    QColor const bgcolor(0xff, 0xff, 0xff);
    OutsidePixels const outside_pixels(OutsidePixels::assumeColor(bgcolor));

    GrayImage dst_bicubic(
        transformToGray(
            src, xform, dst_rect, outside_pixels,
            QSizeF(0.9, 0.9), UPSCALING_BICUBIC_CATMULL_ROM
        )
    );

    // Bicubic upscaling must produce at least one intermediate value
    // strictly between 0 and 255, i.e. a smooth gradient, not a step.
    BOOST_REQUIRE_EQUAL(dst_bicubic.width(), 4);
    bool found_intermediate = false;
    for (int x = 0; x < dst_bicubic.width(); ++x) {
        uint8_t const v = dst_bicubic.data()[x];
        if (v > 0 && v < 255) {
            found_intermediate = true;
            break;
        }
    }
    BOOST_CHECK_MESSAGE(found_intermediate, "Bicubic upscaling produced a hard step, no intermediate values.");
}

BOOST_AUTO_TEST_CASE(test_upscaling_bilinear_smooth_gradient)
{
    int const src_data[] = {0, 255};
    QImage const src = utils::makeGrayImage(src_data, 2, 1);

    QTransform const xform = QTransform().scale(2.0, 1.0);
    QRect const dst_rect(0, 0, 4, 1);
    QColor const bgcolor(0xff, 0xff, 0xff);
    OutsidePixels const outside_pixels(OutsidePixels::assumeColor(bgcolor));

    GrayImage dst_bilinear(
        transformToGray(
            src, xform, dst_rect, outside_pixels,
            QSizeF(0.9, 0.9), UPSCALING_BILINEAR
        )
    );

    // Bilinear interpolation of {0, 255} at positions 0.25, 0.75, 1.25, 1.75
    // (in source coordinates) yields: 255, 64, 191, 255.
    BOOST_REQUIRE_EQUAL(dst_bilinear.width(), 4);
    BOOST_CHECK_EQUAL((int)dst_bilinear.data()[1], 64);
    BOOST_CHECK_EQUAL((int)dst_bilinear.data()[2], 191);
}

BOOST_AUTO_TEST_CASE(test_upscaling_rgb32_smooth_gradient)
{
    // RGB32 image with a black-to-white horizontal transition.
    QImage src(2, 1, QImage::Format_RGB32);
    src.setPixel(0, 0, qRgb(0, 0, 0));
    src.setPixel(1, 0, qRgb(255, 255, 255));

    QTransform const xform = QTransform().scale(2.0, 1.0);
    QRect const dst_rect(0, 0, 4, 1);
    QColor const bgcolor(0xff, 0xff, 0xff);
    OutsidePixels const outside_pixels(OutsidePixels::assumeColor(bgcolor));

    QImage dst_bicubic(
        transform(
            src, xform, dst_rect, outside_pixels,
            QSizeF(0.9, 0.9), UPSCALING_BICUBIC_CATMULL_ROM
        )
    );

    BOOST_REQUIRE_EQUAL(dst_bicubic.width(), 4);
    bool found_intermediate = false;
    for (int x = 0; x < dst_bicubic.width(); ++x) {
        QRgb const px = dst_bicubic.pixel(x, 0);
        int const gray = qGray(px);
        if (gray > 0 && gray < 255) {
            found_intermediate = true;
            break;
        }
    }
    BOOST_CHECK_MESSAGE(found_intermediate, "RGB32 bicubic upscaling produced a hard step, no intermediate values.");
}

BOOST_AUTO_TEST_CASE(test_upscaling_2x_smooth)
{
    // 3x3 gray image with a central black-on-white pattern, upscaled 2x to 6x6.
    int const src_data[] = {
        255, 255, 255,
        255,   0, 255,
        255, 255, 255
    };
    QImage const src = utils::makeGrayImage(src_data, 3, 3);

    QTransform const xform = QTransform().scale(2.0, 2.0);
    QRect const dst_rect(0, 0, 6, 6);
    QColor const bgcolor(0xff, 0xff, 0xff);
    OutsidePixels const outside_pixels(OutsidePixels::assumeColor(bgcolor));

    GrayImage dst(
        transformToGray(
            src, xform, dst_rect, outside_pixels,
            QSizeF(0.9, 0.9), UPSCALING_BICUBIC_CATMULL_ROM
        )
    );

    BOOST_REQUIRE_EQUAL(dst.width(), 6);
    BOOST_REQUIRE_EQUAL(dst.height(), 6);

    // Center of the upscaled image should be dark (region around the black pixel).
    uint8_t const center = dst.data()[3 * dst.stride() + 3];
    BOOST_CHECK_MESSAGE(center < 100, "Upscaled center should be dark, got " << int(center));
}

BOOST_AUTO_TEST_CASE(test_upscaling_identity_preserves_source)
{
    // Verify that identity (no actual upscaling) with bicubic produces
    // an image identical to the source.
    int const src_data[] = {
        10, 60, 110,
        160, 210, 250
    };
    QImage const src = utils::makeGrayImage(src_data, 3, 2);

    QTransform const xform;
    QRect const dst_rect = src.rect();
    QColor const bgcolor(0xff, 0xff, 0xff);
    OutsidePixels const outside_pixels(OutsidePixels::assumeColor(bgcolor));

    GrayImage dst(
        transformToGray(
            src, xform, dst_rect, outside_pixels,
            QSizeF(0.9, 0.9), UPSCALING_BICUBIC_CATMULL_ROM
        )
    );

    BOOST_REQUIRE_EQUAL(dst.width(), src.width());
    BOOST_REQUIRE_EQUAL(dst.height(), src.height());
    // At identity transform, sampling lands exactly on source pixel centers,
    // so Catmull-Rom reproduces the source values (all weights but w_0 vanish).
    for (int y = 0; y < dst.height(); ++y) {
        for (int x = 0; x < dst.width(); ++x) {
            BOOST_CHECK_EQUAL((int)dst.data()[y * dst.stride() + x], src_data[y * 3 + x]);
        }
    }
}

BOOST_AUTO_TEST_CASE(test_downscaling_preserves_area_mapping)
{
    // Downscale 4x1 -> 2x1.  The area-mapping path must be used for
    // downscaling regardless of the chosen upscaling method, so both
    // methods must produce the same (averaged) result.
    int const src_data[] = {0, 0, 255, 255};
    QImage const src = utils::makeGrayImage(src_data, 4, 1);

    QTransform const xform = QTransform().scale(0.5, 1.0);
    QRect const dst_rect(0, 0, 2, 1);
    QColor const bgcolor(0xff, 0xff, 0xff);
    OutsidePixels const outside_pixels(OutsidePixels::assumeColor(bgcolor));

    GrayImage dst_bicubic(
        transformToGray(
            src, xform, dst_rect, outside_pixels,
            QSizeF(0.9, 0.9), UPSCALING_BICUBIC_CATMULL_ROM
        )
    );

    GrayImage dst_box(
        transformToGray(
            src, xform, dst_rect, outside_pixels,
            QSizeF(0.9, 0.9), UPSCALING_BOX
        )
    );

    BOOST_REQUIRE_EQUAL(dst_bicubic.width(), 2);
    BOOST_REQUIRE_EQUAL(dst_box.width(), 2);

    // Downscaling always uses area mapping, so the results agree.
    BOOST_CHECK_EQUAL((int)dst_bicubic.data()[0], (int)dst_box.data()[0]);
    BOOST_CHECK_EQUAL((int)dst_bicubic.data()[1], (int)dst_box.data()[1]);

    // Each destination pixel averages two source pixels: 0+0=0 and 255+255=255.
    // The averaging box also blends the boundary, so expect a mid-tone at
    // the transition rather than a hard step.
    int const total = (int)dst_bicubic.data()[0] + (int)dst_bicubic.data()[1];
    BOOST_CHECK(total > 0 && total < 510);
}

BOOST_AUTO_TEST_SUITE_END();

} // namespace tests

} // namespace imageproc
