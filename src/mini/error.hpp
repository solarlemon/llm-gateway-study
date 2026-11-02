#pragma once
#include <stdexcept>
#include <string>
#include <utility>

// mini 生成错误携带稳定 code 和 HTTP 状态。
struct MiniError : std::runtime_error {
    std::string code;
    int status;
    MiniError(std::string reason, int http_status = 500)
        : std::runtime_error(reason), code(std::move(reason)), status(http_status) {}
};
