#ifndef _COMMON_H_
#define _COMMON_H_

#include <iostream>
#include <sstream>
#include <string>
#include "acl/acl.h"

using ScaleType = std::pair<float, float>;

#define ALIGN_UP(x, size)       ((x + size - 1) & (~(size - 1)))
#define ALIGN16(x)              (ALIGN_UP(x, 16))
#define ALIGN2(x)               (ALIGN_UP(x, 2))

#define CHECK(status) \
{                     \
    if (status != ACL_ERROR_NONE) { \
        fprintf(stderr, "acl error: %d\n", status); \
        abort();\
    }                      \
}

class ThrowErr {
 public:
    ThrowErr() = default;
    explicit ThrowErr(const std::string &msg) {
        ss_ << msg;
    }

    ~ThrowErr() {
        throw std::runtime_error(ss_.str());
    }

    template<class T>
    ThrowErr &operator<<(const T &val) {
        ss_ << val;
        return *this;
    }
 private:
    std::ostringstream ss_;
};

#define CHECK_POINTER(ptr, msg) \
    {                      \
        if (ptr == nullptr) { \
            ThrowErr() << msg;                   \
        }                       \
    }

enum ImgFormat {
    BGR     =   0,
    RGB     =   1,
    NV12    =   2,
    NV21    =   3,
};

struct ImgBuffer {
    uint32_t width;
    uint32_t height;
    ImgFormat format;
    void *data;
};

#endif  // _COMMON_H_
