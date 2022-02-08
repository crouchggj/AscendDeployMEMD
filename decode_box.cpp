#include "decode_box.h"
#include "acl/acl.h"

float doIou(const cv::Rect &a, const cv::Rect &b) {
    cv::Rect inter =  a | b;
    cv::Rect combine = a & b;
    return static_cast<float>(combine.area()) / inter.area();
}

std::vector<BBox> doNms(std::vector<BBox> &bbox, float nms) {
    std::sort(bbox.begin(), bbox.end(), [](const BBox &a, const BBox &b) {
        return a.score > b.score;
    });

    std::vector<BBox> result;
    std::vector<bool> available(bbox.size(), true);
    for (size_t i = 0; i < bbox.size(); i++) {
        if (available[i] == false) continue;

        result.push_back(bbox[i]);
        for (size_t j = i + 1; j < bbox.size(); j++) {
            if (available[j] == false) continue;
            float iou = doIou(bbox[i].rect, bbox[j].rect);
            if (iou > nms) {
                available[j] = false;
            }
        }
    }
    return result;
}

std::vector<BBox> DecodeBBox(const std::vector<ImgBuffer> & output_buffer, const ScaleType scales,
                             float confidence, float nms) {
    std::vector<float> confidence_feature(output_buffer[0].width * output_buffer[0].height);
    std::vector<float> box_feature(output_buffer[1].width * output_buffer[1].height);
    aclrtMemcpy(confidence_feature.data(), confidence_feature.size() * sizeof(float),
                output_buffer[0].data, confidence_feature.size() * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(box_feature.data(), box_feature.size() * sizeof(float),
                output_buffer[1].data, box_feature.size() * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<BBox> filter_bboxes;
    for (uint32_t i = 0; i < confidence_feature.size(); i++) {
        if (confidence_feature[i] > confidence) {
            int xmin = box_feature[i * 4 + 0] / scales.first;
            int ymin = box_feature[i * 4 + 1] / scales.second;
            int xmax = box_feature[i * 4 + 2] / scales.first;
            int ymax = box_feature[i * 4 + 3] / scales.second;
            BBox bbox {
                cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)),
                confidence_feature[i]
            };
            filter_bboxes.emplace_back(bbox);
        }
    }

    return doNms(filter_bboxes, nms);
}