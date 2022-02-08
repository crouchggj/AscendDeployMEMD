#include "acl_dataset.h"

void __InitPicDesc(ImgFormat format, int width, int height, void *data_buffer, acldvppPicDesc *pic_desc) {
    int out_buffer_size = 0;
    if (format == BGR) {
        out_buffer_size = width * height * 3;
        acldvppSetPicDescFormat(pic_desc, PIXEL_FORMAT_BGR_888);
        acldvppSetPicDescWidthStride(pic_desc, width * 3);
        acldvppSetPicDescHeightStride(pic_desc, height);
    } else if (format == RGB) {
        out_buffer_size = width * height * 3;
        acldvppSetPicDescFormat(pic_desc, PIXEL_FORMAT_RGB_888);
        acldvppSetPicDescWidthStride(pic_desc, width * 3);
        acldvppSetPicDescHeightStride(pic_desc, height);
    } else if (format == NV12) {
        out_buffer_size = width * height * 3 / 2;
        acldvppSetPicDescFormat(pic_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
        acldvppSetPicDescWidthStride(pic_desc, width);
        acldvppSetPicDescHeightStride(pic_desc, height);
    } else if (format == NV21) {
        out_buffer_size = width * height * 3 / 2;
        acldvppSetPicDescFormat(pic_desc, PIXEL_FORMAT_YVU_SEMIPLANAR_420);
        acldvppSetPicDescWidthStride(pic_desc, width);
        acldvppSetPicDescHeightStride(pic_desc, height);
    } else {
        ThrowErr() << "Image format has not been supported";
    }

    acldvppSetPicDescData(pic_desc, data_buffer);
    acldvppSetPicDescWidth(pic_desc, width);
    acldvppSetPicDescHeight(pic_desc, height);
    acldvppSetPicDescSize(pic_desc, out_buffer_size);
}


void AclDataset::createDvppDesc() {
    dvpp_channel_desc_ = acldvppCreateChannelDesc();
    CHECK_POINTER(dvpp_channel_desc_, "create channel desc failed");

    CHECK(acldvppCreateChannel(dvpp_channel_desc_));

    resize_config_ = acldvppCreateResizeConfig();
    CHECK_POINTER(resize_config_, "create resize config failed");

    CHECK(aclrtCreateStream(&stream_));
}

void AclDataset::destroyDvppDesc() {
    if (resize_config_ != nullptr) {
        acldvppDestroyResizeConfig(resize_config_);
        resize_config_ = nullptr;
    }

    if (dvpp_channel_desc_ != nullptr) {
        acldvppDestroyChannel(dvpp_channel_desc_);
        acldvppDestroyChannelDesc(dvpp_channel_desc_);
        dvpp_channel_desc_ = nullptr;
    }

    if (stream_ != nullptr) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
}

void AclDataset::createMdDesc() {
    model_desc_ = aclmdlCreateDesc();
    CHECK_POINTER(model_desc_, "create md desc failed");

    CHECK(aclmdlGetDesc(model_desc_, model_id_));

    input_ = aclmdlCreateDataset();
    CHECK_POINTER(input_, "create input dataset failed");

    size_t input_size = aclmdlGetNumInputs(model_desc_);
    for (size_t i = 0; i < input_size; ++i) {
        aclmdlIODims dims;
        aclmdlGetInputDims(model_desc_, i, &dims);
        size_t buffer_size = aclmdlGetInputSizeByIndex(model_desc_, i);
        void *input_buffer = nullptr;
        CHECK(acldvppMalloc(&input_buffer, buffer_size));

        aclDataBuffer *input_data = aclCreateDataBuffer(input_buffer, buffer_size);
        CHECK_POINTER(input_data, "create input data buffer failed")
        CHECK(aclmdlAddDatasetBuffer(input_, input_data));
        ImgBuffer buffer;
        buffer.width = dims.dims[2];
        buffer.height = dims.dims[1];
        buffer.data = aclGetDataBufferAddr(input_data);
        input_buffer_.emplace_back(buffer);
    }

    output_ = aclmdlCreateDataset();
    CHECK_POINTER(output_, "create output dataset failed");

    size_t output_size = aclmdlGetNumOutputs(model_desc_);
    for (size_t i = 0; i < output_size; ++i) {
        size_t buffer_size = aclmdlGetOutputSizeByIndex(model_desc_, i);

        void *output_buffer = nullptr;
        CHECK(aclrtMalloc(&output_buffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY));

        aclDataBuffer *output_data = aclCreateDataBuffer(output_buffer, buffer_size);
        CHECK_POINTER(output_data, "create output buffer failed");
        CHECK(aclmdlAddDatasetBuffer(output_, output_data));
        aclmdlIODims dims;
        aclmdlGetOutputDims(model_desc_, i, &dims);
        ImgBuffer buffer;
        buffer.width = dims.dims[0];
        buffer.height = dims.dims[1];
        buffer.data = aclGetDataBufferAddr(output_data);
        output_buffer_.emplace_back(buffer);
    }
}

void AclDataset::destroyMdDesc() {
    if (model_desc_ != nullptr) {
        (void) aclmdlDestroyDesc(model_desc_);
        model_desc_ = nullptr;
    }

    if (output_) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
            aclDataBuffer *data_buffer = aclmdlGetDatasetBuffer(output_, i);
            void *data = aclGetDataBufferAddr(data_buffer);
            (void) aclrtFree(data);
            (void) aclDestroyDataBuffer(data_buffer);
        }

        (void) aclmdlDestroyDataset(output_);
        output_ = nullptr;
    }

    if (input_) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
            aclDataBuffer *data_buffer = aclmdlGetDatasetBuffer(input_, i);
            void *data = aclGetDataBufferAddr(data_buffer);
            (void) acldvppFree(data);
            aclDestroyDataBuffer(data_buffer);
        }

        aclmdlDestroyDataset(input_);
        input_ = nullptr;
    }
}

AclDataset::AclDataset(uint32_t model_id) {
    model_id_ = model_id;
    createDvppDesc();
    createMdDesc();
}

AclDataset::~AclDataset() {
    destroyDvppDesc();
    destroyMdDesc();
}
