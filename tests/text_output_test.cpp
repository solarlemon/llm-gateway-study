#include "../src/text_output.hpp"
#include <iostream>
void require(bool ok)
{
    if (!ok)
        throw std::runtime_error("text_output assertion failed");
}
int main()
{
    std::string result;
    TextOutput utf8({}, [&](const auto &s)
                    { result += s; });
    utf8.push("\xe4");
    require(result.empty());
    utf8.push("\xbd");
    require(result.empty());
    utf8.push("\xa0!");
    utf8.finish();
    require(result == "你!");
    result.clear();
    TextOutput stop({"END"}, [&](const auto &s)
                    { result += s; });
    require(!stop.push("abcE"));
    require(result == "abc");
    require(stop.push("NDhidden"));
    stop.finish();
    require(result == "abc");
    result.clear();
    TextOutput incomplete({}, [&](const auto &s)
                          { result += s; });
    incomplete.push("\xe4");
    incomplete.finish();
    require(result == "\xef\xbf\xbd");
    result.clear();
    TextOutput unmatched({"END"}, [&](const auto &s)
                         { result += s; });
    unmatched.push("abcEN");
    unmatched.finish();
    require(result == "abcEN");
    std::cout << "PASS UTF8 boundaries, split stop, incomplete character, unmatched stop prefix\n";
}
