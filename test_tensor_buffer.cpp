#include "ranges.hpp"

#include <format>
#include <print>
#include <string>

#include "tensor_buffer.hpp"

int
main() {
    // rank 0
    auto scalar_buff = tensor_buffer<0, 4, double, std::allocator<double>>(2, 2, 2);

    scalar_buff.for_each_index([&](const grid_index idx, const tensor_index<0> tidx) {
        const auto [i, j, k]       = idx;
        scalar_buff[i, j, k][tidx] = static_cast<double>(100 * i + 10 * j + k);
    });

    scalar_buff.for_each_index([&](const grid_index idx, const tensor_index<0> tidx) {
        const auto [i, j, k] = idx;
        std::println("[{}, {}, {}][] = {:3}", i, j, k, scalar_buff[idx][tidx]);
    });

    std::println();

    // rank 2
    auto buff = tensor_buffer<2, 3, std::string, std::allocator<std::string>>(3, 1, 2);

    // writing
    buff.for_each_index([&](const grid_index idx, const tensor_index<2> tidx) {
        const auto [i, j, k] = idx;
        const auto [x, y] = tidx;
        buff[idx][tidx] = std::format("[{}, {}, {}] [{}, {}]", i, j, k, x, y);
    });

    // reading const buff
    const auto& buff_const_ref = buff;

    buff_const_ref.for_each_index([&](const grid_index idx, const tensor_index<2> tidx) {
        const auto [i, j, k] = idx;
        const auto [x, y] = tidx;
        std::println("[{}, {}, {}] [{}, {}] = {}", i, j, k, x, y, buff[idx][tidx]);
    });
}
