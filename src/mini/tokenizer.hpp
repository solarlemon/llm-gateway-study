#pragma once
#include "model.hpp"
#include <string>
#include <unordered_map>

namespace mini {
// 配套 tok512.bin 的 SentencePiece-style BPE，词表构造完成后只读，可供 worker 共享。
class Tokenizer {
public:
    Tokenizer(const std::string& path, int vocabulary_size);
    std::vector<TokenId> encode(const std::string& text, const Cancel& cancel = {}) const;
    std::string piece(TokenId previous, TokenId token) const;
private:
    std::vector<std::string> vocabulary_;
    std::vector<float> scores_;
    std::unordered_map<std::string,TokenId> lookup_;
};
}
