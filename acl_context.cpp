#include "acl_context.h"

ModelType::ModelType(const char *model_path) {
    CHECK(aclmdlQuerySize(model_path,
                          &model_mem_size_, &model_weight_size_));

    CHECK(aclrtMalloc(&model_mem_ptr_, model_mem_size_, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&model_weight_ptr_, model_weight_size_, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK(aclmdlLoadFromFileWithMem(model_path, &model_id_,
                                    model_mem_ptr_, model_mem_size_,
                                    model_weight_ptr_, model_weight_size_));
    fprintf(stdout, "Load model success %s", model_path);
}

ModelType::~ModelType() {
    if (aclmdlUnload(model_id_) == ACL_ERROR_NONE) {
        aclrtFree(model_mem_ptr_);
        aclrtFree(model_weight_ptr_);
    }
}

AclContext::AclContext(int device_id) : device_id_(device_id) {
    CHECK(aclInit(NULL));
    CHECK(aclrtSetDevice(device_id));
}

bool AclContext::loadModel(const std::string &path) {
    model_ptr_ = std::make_unique<ModelType>(path.c_str());
    dataset_ = std::make_unique<AclDataset>(model_ptr_->id());
    return true;
}

bool AclContext::doInference() {
    if (!model_ptr_ || !dataset_) {
        fprintf(stderr, "load model first");
        return false;
    }

    aclError r = aclmdlExecute(model_ptr_->id(), dataset_->input(), dataset_->output());
    if (r != ACL_ERROR_NONE) {
        fprintf(stderr, "inference failure, error: %d\n", r);
        return false;
    }
    return true;
}

void AclContext::unloadModel() {
    dataset_.reset(nullptr);
    model_ptr_.reset(nullptr);
}

AclContext::~AclContext() {
    CHECK(aclrtResetDevice(device_id_));
    CHECK(aclFinalize());
}
