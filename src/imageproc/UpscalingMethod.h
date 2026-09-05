/*
    Scan Tailor - Interactive post-processing tool for scanned pages.
    Copyright (C) 2026 ScanTailor Deviant Contributors

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

#ifndef IMAGEPROC_UPSCALING_METHOD_H_
#define IMAGEPROC_UPSCALING_METHOD_H_

#include <QString>

namespace imageproc
{

enum UpscalingMethod
{
    UPSCALING_AUTO = 0,
    UPSCALING_BICUBIC_CATMULL_ROM,
    UPSCALING_BICUBIC_MITCHELL,
    UPSCALING_BILINEAR,
    UPSCALING_BOX
};

inline UpscalingMethod upscalingMethodFromString(QString const& str)
{
    QString const s = str.trimmed().toLower();
    if (s == "mitchell" || s == "bicubic_mitchell" || s == "mitchell-netravali") {
        return UPSCALING_BICUBIC_MITCHELL;
    } else if (s == "bilinear") {
        return UPSCALING_BILINEAR;
    } else if (s == "box" || s == "nearest" || s == "legacy") {
        return UPSCALING_BOX;
    }
    return UPSCALING_BICUBIC_CATMULL_ROM;
}

inline QString upscalingMethodToString(UpscalingMethod method)
{
    switch (method) {
    case UPSCALING_BICUBIC_MITCHELL:
        return "mitchell";
    case UPSCALING_BILINEAR:
        return "bilinear";
    case UPSCALING_BOX:
        return "box";
    case UPSCALING_BICUBIC_CATMULL_ROM:
    default:
        return "bicubic";
    }
}

UpscalingMethod defaultUpscalingMethod();
void setDefaultUpscalingMethod(UpscalingMethod method);

} // namespace imageproc

#endif
