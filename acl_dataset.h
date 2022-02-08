#ifndef _ACL_DATASET_H_
#define _ACL_DATASET_H_

#include "acl/ops/acl_dvpp.h"
#include <vector>
#include "common.h"

class AclDataset {
 public:
    explicit AclDataset(uint32_t);
    ~AclDataset();

    aclmdlDataset * input() {
        return input_;
    }

    aclmdlDataset * output() {
        return output_;
    }

    std::vector<ImgBuffer> & inputBuffer() {
        return input_buffer_;
    }

    std::vector<ImgBuffer> & outputBuffer() {
        return output_buffer_;
    }

 private:
    void createDvppDesc();
    void destroyDvppDesc();
    void createMdDesc();
    void destroyMdDesc();

 private:
    uint32_t model_id_;
    std::vector<ImgBuffer> input_buffer_;
    std::vector<ImgBuffer> output_buffer_;
    aclmdlDesc *model_desc_;
    aclmdlDataset *input_;
    aclmdlDataset *output_;
    acldvppChannelDesc *dvpp_channel_desc_;
    acldvppResizeConfig *resize_config_;
    aclrtStream stream_;
};

#endif  // _ACL_DATASET_H_
