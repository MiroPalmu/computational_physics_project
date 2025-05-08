#include "ranges.hpp"

#include <format>
#include <print>
#include <memory>
#include <string>

#include <sycl/sycl.hpp>

#include "idg/sstd.hpp"

#include "tensor_buffer.hpp"
#include "grid_types.hpp"

sycl::queue tensor_buffer_queue =
    sycl::queue(sycl::gpu_selector_v, { sycl::property::queue::in_order{} });

using allocator = sycl::usm_allocator<real, sycl::usm::alloc::shared>;

int
main() {
    struct Foo {
        tensor_buffer<0, 4, real, allocator> scalar_buff{2, 2, 2};

        void operator()() {
            using buff_type = tensor_buffer<0, 4, real, allocator>;
            auto reverse_scalar_buff_ptr = std::allocate_shared<buff_type>(allocator{tensor_buffer_queue}, 2, 2, 2);
            scalar_buff.for_each_index([this, SPTR(reverse_scalar_buff_ptr)](const grid_index idx, const tensor_index<0> tidx) {
                const auto [i, j, k]             = idx;
                scalar_buff[i, j, k][tidx] = static_cast<real>(100 * i + 10 * j + k);
                (*reverse_scalar_buff_ptr)[i, j, k][tidx] = static_cast<real>(i + 10 * j + 100 * k);
            });

            tensor_buffer_queue.wait();

            for (const auto idx : idg::sstd::geometric_index_space<3, 2>()) {
                std::println("{}, {}, {} : {}", idx[0], idx[1], idx[2], scalar_buff[idx][]);
                std::println("{}, {}, {} : {}", idx[0], idx[1], idx[2], (*reverse_scalar_buff_ptr)[idx][]);
            }
        }

    };

    auto f = std::allocate_shared<Foo>(allocator{tensor_buffer_queue});
    (*f)();

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
