#include "ranges.hpp"

#include <format>
#include <print>
#include <string>

#include <sycl/sycl.hpp>

#include "idg/sstd.hpp"

#include "tensor_buffer.hpp"

sycl::queue tensor_buffer_queue =
    sycl::queue(sycl::cpu_selector_v, { sycl::property::queue::in_order{} });

int
main() {
    struct Foo {
        tensor_buffer<0, 4, double, std::allocator<double>> scalar_buff{ 2, 2, 2 };

        void operator()() {
            scalar_buff.for_each_index([this](const grid_index idx, const tensor_index<0> tidx) {
                const auto [i, j, k]             = idx;
                scalar_buff[i, j, k][tidx] = static_cast<double>(100 * i + 10 * j + k);
            });

            tensor_buffer_queue.wait();

            for (const auto idx : idg::sstd::geometric_index_space<3, 2>()) {
                std::println("{}, {}, {} : {}", idx[0], idx[1], idx[2], scalar_buff[idx][]);
            }
        }
    };

    auto f = Foo{};
    f();

    /*

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
        const auto [x, y]    = tidx;
        buff[idx][tidx]      = std::format("[{}, {}, {}] [{}, {}]", i, j, k, x, y);
    });

    // reading const buff
    const auto& buff_const_ref = buff;

    buff_const_ref.for_each_index([&](const grid_index idx, const tensor_index<2> tidx) {
        const auto [i, j, k] = idx;
        const auto [x, y]    = tidx;
        std::println("[{}, {}, {}] [{}, {}] = {}", i, j, k, x, y, buff[idx][tidx]);
    });

    buff_const_ref.for_each_index([&](const grid_index idx) {
        const auto [i, j, k] = idx;
        std::println("[{}, {}, {}] [0,1] and [1, 1] = {} {}",
                     i,
                     j,
                     k,
                     buff[idx][0, 1],
                     buff[idx][1, 1]);
    });
    */
}
