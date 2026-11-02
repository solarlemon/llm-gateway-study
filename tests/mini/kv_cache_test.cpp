#include "kv_cache.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

void require(bool condition) {
    if (!condition) throw std::runtime_error("KV test assertion failed");
}
template<class Exception, class Action> void rejects(Action action) {
    try { action(); } catch (const Exception&) { return; }
    throw std::runtime_error("Expected KV exception was not raised");
}

int main() {
    using mini::KVCache;
    rejects<std::invalid_argument>([] { KVCache c({0, 2, 2, 3}); });
    rejects<std::overflow_error>([] { KVCache c({2, std::numeric_limits<std::size_t>::max(), 2, 3}); });
    rejects<std::overflow_error>([] { KVCache c({1, std::numeric_limits<std::size_t>::max()/sizeof(float), 1, 1}); });
    KVCache budget({5, 512, 4, 8});
    require(budget.allocated_bytes() == 655360 && budget.valid_bytes() == 0);

    KVCache a({2, 2, 2, 3}), b({2, 2, 2, 3});
    const std::vector<float> k{1,2,3,4,5,6}, v{11,12,13,14,15,16};
    rejects<std::out_of_range>([&] { a.key(0,0,0,0); });
    rejects<std::invalid_argument>([&] { a.write_layer(0, {1}, v); });
    rejects<std::out_of_range>([&] { a.write_layer(2, k, v); });
    a.write_layer(0,k,v);
    require(a.valid_len() == 0 && a.key(0,0,1,2) == 6 && a.value(0,0,1,2) == 16);
    rejects<std::out_of_range>([&] { a.key(1,0,0,0); });
    rejects<std::out_of_range>([&] { a.key(0,1,0,0); });
    rejects<std::out_of_range>([&] { a.key(0,0,2,0); });
    rejects<std::out_of_range>([&] { a.key(0,0,0,3); });
    rejects<std::logic_error>([&] { a.commit_token(); });
    rejects<std::logic_error>([&] { a.write_layer(0,k,v); });
    a.write_layer(1,v,k);
    a.commit_token();
    require(a.valid_len() == 1 && a.valid_bytes() == 96);
    for (std::size_t h=0; h<2; ++h) for (std::size_t d=0; d<3; ++d) {
        require(a.key(0,0,h,d) == k[h*3+d]);
        require(a.value(1,0,h,d) == k[h*3+d]);
    }
    b.write_layer(0,v,k); b.write_layer(1,v,k); b.commit_token();
    require(a.key(0,0,0,0) == 1 && b.key(0,0,0,0) == 11);
    a.write_layer(0,v,k); a.rollback_token();
    require(a.valid_len() == 1 && a.key(0,0,0,0) == 1);
    rejects<std::out_of_range>([&] { a.key(0,1,0,0); });
    a.write_layer(0,v,k); a.write_layer(1,k,v); a.commit_token();
    require(a.valid_len() == 2 && a.valid_bytes() == a.allocated_bytes());
    require(a.key(0,0,0,0) == 1 && a.key(0,1,0,0) == 11);
    rejects<std::out_of_range>([&] { a.write_layer(0,k,v); });
    rejects<std::out_of_range>([&] { a.commit_token(); });
    a.reset(); a.reset();
    require(a.valid_len() == 0 && a.valid_bytes() == 0);
    rejects<std::out_of_range>([&] { a.key(0,0,0,0); });
    require(b.key(0,0,0,0) == 11);
    a.write_layer(0,k,v); a.write_layer(1,k,v); a.commit_token();
    require(a.key(0,0,1,2) == 6);
    rejects<std::out_of_range>([&] { a.value(1,1,0,0); });
    std::cout << "PASS KV layout, pending reads, commit, rollback, reset, isolation, capacity, overflow\n";
}
