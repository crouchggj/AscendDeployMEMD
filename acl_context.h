#ifndef _ACL_CONTEXT_H_
#define _ACL_CONTEXT_H_

#include <string>
#include <memory>
#include <iostream>
#include "common.h"
#include "acl_dataset.h"

class ModelType {
 public:
    explicit ModelType(const char *model_path);
    ~ModelType();

    uint32_t id() {
        return model_id_;
    }

 private:
    uint32_t model_id_;
    size_t model_mem_size_;
    size_t model_weight_size_;
    void *model_mem_ptr_;
    void *model_weight_ptr_;
};

class AclContext {
 public:
    explicit AclContext(int device_id);
    ~AclContext();

    bool loadModel(const std::string & path);
    bool doInference();
    void unloadModel();

    std::unique_ptr<AclDataset> & getDataset() {
        return dataset_;
    }

 private:
    const int device_id_;
    std::unique_ptr<ModelType> model_ptr_;
    std::unique_ptr<AclDataset> dataset_;
};

#endif  // _ACL_CONTEXT_H_
