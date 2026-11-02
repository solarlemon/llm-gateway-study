// BPE and byte-piece conventions follow llama2.c (MIT), see THIRD_PARTY.md.
#include "tokenizer.hpp"
#include "../text_output.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace mini {
Tokenizer::Tokenizer(const std::string& path, int vocabulary_size) {
    if (vocabulary_size < 259 || vocabulary_size > 65536)
        throw std::invalid_argument("unsupported tokenizer vocabulary");
    std::ifstream input(path,std::ios::binary);
    std::uint32_t maximum = 0;
    input.read(reinterpret_cast<char*>(&maximum),sizeof(maximum));
    if (!input || maximum == 0 || maximum > 4096) throw std::runtime_error("invalid tokenizer header");
    for (int i=0; i<vocabulary_size; ++i) {
        float score = 0;
        std::int32_t length = 0;
        input.read(reinterpret_cast<char*>(&score),sizeof(score));
        input.read(reinterpret_cast<char*>(&length),sizeof(length));
        if (!input || !std::isfinite(score) || length <= 0 || length > static_cast<int>(maximum))
            throw std::runtime_error("invalid tokenizer entry");
        std::string word(length,'\0');
        input.read(word.data(),length);
        if (!input || word.find('\0') != std::string::npos || !lookup_.emplace(word,i).second)
            throw std::runtime_error("invalid tokenizer vocabulary");
        vocabulary_.push_back(std::move(word)); scores_.push_back(score);
    }
    if (input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("tokenizer trailing bytes");
    if (!lookup_.count(" ")) throw std::runtime_error("tokenizer missing space prefix");
    for (int i=0; i<256; ++i) {
        std::ostringstream expected;
        expected << "<0x" << std::uppercase << std::hex << std::setw(2) << std::setfill('0') << i << ">";
        if (vocabulary_[i+3] != expected.str()) throw std::runtime_error("tokenizer byte fallback mismatch");
    }
}

std::vector<TokenId> Tokenizer::encode(const std::string& text, const Cancel& cancel) const {
    if (text.size() > 8192 || text.find('\0') != std::string::npos)
        throw std::invalid_argument("mini prompt must be at most 8192 bytes and contain no NUL");
    // 使用同一 UTF-8 校验器拒绝不完整字符和非法字节。
    std::string validated;
    TextOutput check({},[&](const std::string& s){ validated += s; });
    check.push(text); check.finish();
    if (validated != text) throw std::invalid_argument("invalid prompt UTF-8");
    std::vector<TokenId> ids{1}; // BOS。
    if (!text.empty()) ids.push_back(lookup_.at(" "));
    for (std::size_t i=0; i<text.size();) {
        const auto byte = static_cast<unsigned char>(text[i]);
        const std::size_t length = byte < 0x80 ? 1 : byte < 0xe0 ? 2 : byte < 0xf0 ? 3 : 4;
        const auto found = lookup_.find(text.substr(i,length));
        if (found != lookup_.end()) ids.push_back(found->second);
        else for (std::size_t j=0; j<length; ++j) ids.push_back(static_cast<unsigned char>(text[i+j])+3);
        i += length;
    }
    for (;;) {
        if (cancel && cancel()) throw std::runtime_error("mini_cancelled");
        float best_score = -1e10f;
        TokenId best_id = -1;
        std::size_t best_index = 0;
        for (std::size_t i=0; i+1<ids.size(); ++i) {
            const auto found = lookup_.find(vocabulary_[ids[i]]+vocabulary_[ids[i+1]]);
            if (found != lookup_.end() && scores_[found->second] > best_score) {
                best_score = scores_[found->second]; best_id = found->second; best_index = i;
            }
        }
        if (best_id < 0) break;
        ids[best_index] = best_id;
        ids.erase(ids.begin()+best_index+1);
    }
    return ids;
}

std::string Tokenizer::piece(TokenId previous, TokenId token) const {
    if (token < 0 || static_cast<std::size_t>(token) >= vocabulary_.size())
        throw std::out_of_range("tokenizer ID out of range");
    auto value = vocabulary_[token];
    if (previous == 1 && !value.empty() && value.front() == ' ') value.erase(0,1);
    if (token >= 3 && token <= 258) return std::string(1,static_cast<char>(token-3));
    return value;
}
}
