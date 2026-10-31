#pragma once
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

// 文本层安全缓冲器：持有可能的停止串前缀和不完整 UTF-8 字节，
// 只向外输出"完整、合法、且不可能是停止串前缀"的文本。
// native 路径的 token piece 不保证是完整字符，故需要本类做边界裁剪。
class TextOutput
{
    std::string pending;                             // 尚未安全输出的累积缓冲
    std::vector<std::string> stops;                  // 停止串集合（任一匹配即停止）
    std::function<void(const std::string &)> emit;   // 安全文本的唯一出口
    bool stopped = false;                            // 是否已匹配停止串（置位后不再输出）

    // 逐字节扫描，返回 limit 内完整且合法的 UTF-8 字节边界；不完整尾部留给下次，非法编码抛异常。
    // 严格校验：拒绝 overlong、surrogate、超出 U+10FFFF 的编码。
    static size_t complete_utf8(const std::string &s, size_t limit)
    {
        size_t i = 0;
        while (i < limit)
        {
            const auto c = static_cast<unsigned char>(s[i]);
            // 按首字节判断字符长度；C0/C1、F5-FF 等非法首字节直接判 0
            size_t n = c < 0x80 ? 1
                     : c >= 0xc2 && c <= 0xdf ? 2
                     : c >= 0xe0 && c <= 0xef ? 3
                     : c >= 0xf0 && c <= 0xf4 ? 4
                     : 0;
            if (!n)
                throw std::runtime_error("invalid UTF-8 token piece");
            if (i + n > limit)
                break;                                // 字符跨 limit 边界，留待下次
            // 后续字节必须是 10xxxxxx
            for (size_t k = 1; k < n; ++k)
                if ((static_cast<unsigned char>(s[i + k]) & 0xc0) != 0x80)
                    throw std::runtime_error("invalid UTF-8 continuation");
            // 3/4 字节字符的 overlong / surrogate / 越界校验
            if (n >= 3)
            {
                const auto next = static_cast<unsigned char>(s[i + 1]);
                if ((c == 0xe0 && next < 0xa0) || (c == 0xed && next >= 0xa0) ||
                    (c == 0xf0 && next < 0x90) || (c == 0xf4 && next >= 0x90))
                    throw std::runtime_error("invalid UTF-8 scalar");
            }
            i += n;
        }
        return i;                                     // i 之前均为完整合法 UTF-8
    }

    // 排出 pending 中前 limit 字节内的完整合法文本，剩余留在 pending
    void drain(size_t limit)
    {
        const size_t n = complete_utf8(pending, limit);
        if (n)
        {
            emit(pending.substr(0, n));               // 只发完整合法前缀
            pending.erase(0, n);                      // 从缓冲移除已发部分
        }
    }

public:
    TextOutput(std::vector<std::string> patterns, std::function<void(const std::string &)> callback)
        : stops(std::move(patterns)), emit(std::move(callback)) {}

    // 接收一个 token piece。返回 true 表示已匹配停止串（调用方可提前终止生成）。
    bool push(const std::string &piece)
    {
        if (stopped)
            return true;                              // 已停止，忽略后续
        pending += piece;

        // 1) 检查 pending 中是否已完整出现停止串（可能跨多个 piece）
        size_t match = std::string::npos;
        for (const auto &s : stops)
            match = std::min(match, pending.find(s));
        if (match != std::string::npos)
        {
            drain(match);                             // 停止串之前的内容安全输出
            pending.clear();                          // 停止串及其后全部丢弃
            stopped = true;
            return true;
        }

        // 2) 找出必须保留的最长"可能是停止串前缀"的后缀
        //    例：stop="END"，pending 尾部="EN"，必须保留，否则跨片停止串无法识别
        size_t keep = 0;
        for (const auto &s : stops)
        {
            for (size_t k = 1; k < s.size() && k <= pending.size(); ++k)
                if (pending.compare(pending.size() - k, k, s, 0, k) == 0)
                    keep = std::max(keep, k);
        }

        // 3) 排出除保留后缀外的全部内容（drain 还会再受 UTF-8 边界约束）
        drain(pending.size() - keep);
        return false;                                 // 未停止，继续
    }

    // 正常结束时收尾：排出剩余完整文本；若只剩半个字符，输出 U+FFFD 替代。
    // 已匹配停止串则不输出任何残留。
    void finish()
    {
        if (stopped)
            return;
        drain(pending.size());                        // 排出所有完整合法文本
        // max_tokens 截断可能切断 UTF-8 字符，发替代字符保证输出合法
        if (!pending.empty())
        {
            emit("\xef\xbf\xbd");                     // U+FFFD REPLACEMENT CHARACTER
            pending.clear();
        }
    }
};