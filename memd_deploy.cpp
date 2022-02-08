#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>
#include <string>
#include <utility>
#include <optional>
#include "acl_context.h"
#include "decode_box.h"

namespace {
constexpr float kConfidence = 0.5;
constexpr float kNMS = 0.6;
}

void helper() {
    fprintf(stdout, "./AscendDeployMEMD memd.om test.jpg");
    exit(0);
}

std::optional<ScaleType> resize(const std::string jpg_path, std::unique_ptr<AclContext> &session) {
    cv::Mat image = cv::imread(jpg_path);
    auto &model_input_buffer = session->getDataset()->inputBuffer();
    fprintf(stdout, "model size: %d x %d\n", model_input_buffer[0].width, model_input_buffer[0].height);
    // 非等比例缩放
    float scale_w = static_cast<float>(model_input_buffer[0].width) / image.cols;
    float scale_h = static_cast<float>(model_input_buffer[0].height) / image.rows;
    cv::Mat target_img;
    cv::resize(image, target_img, cv::Size(0, 0), scale_w, scale_h);

    aclError r = aclrtMemcpy(model_input_buffer[0].data, target_img.total() * target_img.channels(),
                             target_img.data, target_img.total() * target_img.channels(),
                             ACL_MEMCPY_HOST_TO_DEVICE);
    if (r != ACL_ERROR_NONE) {
        fprintf(stderr, "acl memcpy failed\n");
        return std::nullopt;
    }

    return std::make_pair(scale_w, scale_h);
}

void draw(const std::string jpg_path, const std::vector<BBox> &bboxes) {
    cv::Mat image = cv::imread(jpg_path);
    for (auto &m : bboxes) {
        cv::rectangle(image, m.rect.tl(), m.rect.br(), cv::Scalar(0, 0, 255), 2);
        cv::putText(image, std::to_string(m.score),
                    cv::Point(m.rect.x + m.rect.width / 2, m.rect.y + m.rect.height / 2),
                    1, 0.8, cv::Scalar(255, 0, 255));
        fprintf(stdout, "output [%d %d %d %d] score: %f\n",
                m.rect.x, m.rect.y, m.rect.width, m.rect.height, m.score);
    }
    cv::imwrite("memd_result.jpg", image);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        helper();
    }

    auto session = std::make_unique<AclContext>(0);
    std::string model_path = argv[1];
    std::string jpg_path = argv[2];

    session->loadModel(model_path);

    auto input_scale = resize(jpg_path, session);
    if (input_scale) {
        auto t0 = std::chrono::steady_clock::now();
        bool r = session->doInference();
        if (!r) {
            abort();
        }
        auto t1 = std::chrono::steady_clock::now();
        fprintf(stdout, "Inference Success, cost time: %ld ms\n",
                std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
        auto &output_buffer = session->getDataset()->outputBuffer();
        auto result_bboxes = DecodeBBox(output_buffer, input_scale.value(),
                                        kConfidence, kNMS);
        draw(jpg_path, result_bboxes);
    }

    session->unloadModel();
    fprintf(stdout, "Finish Success\n");
    return 0;
}
