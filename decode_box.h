#ifndef _DECODE_BOX_H_
#define _DECODE_BOX_H_

#include <opencv2/core.hpp>
#include <vector>
#include "common.h"

struct BBox {
    cv::Rect rect;
    float score;
};

std::vector<BBox> DecodeBBox(const std::vector<ImgBuffer> & output_buffer, const ScaleType scales,
                             float confidence, float nms);

#endif  // _DECODE_BOX_H_
