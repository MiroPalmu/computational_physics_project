#include "ranges.hpp"

#include <format>
#include <print>
#include <string>

#include "tensor_buffer.hpp"

int
main() {
    // rank 0
    auto scalar_buff = tensor_buffer<0, 4, double, std::allocator<double>>(2, 2, 2);

    for (const auto [i, j, k] :
         rv::cartesian_product(rv::iota(0uz, 2uz), rv::iota(0uz, 2uz), rv::iota(0uz, 2uz))) {
        auto mds_at_ijk = scalar_buff[i, j, k];

        // with array
        mds_at_ijk[std::array<std::size_t, 0>{}] = 100 * i + 10 * j + k;

        // wout array
        std::println("[{}, {}, {}][] = {:3}", i, j, k, mds_at_ijk[]);
    }

    std::println();

    // rank 2
    auto buff = tensor_buffer<2, 3, std::string, std::allocator<std::string>>(3, 1, 2);

    // writing
    for (const auto [i, j, k] :
         rv::cartesian_product(rv::iota(0uz, 3uz), rv::iota(0uz, 1uz), rv::iota(0uz, 2uz))) {
        auto mds_at_ijk = buff[i, j, k];

        for (const auto [x, y] : rv::cartesian_product(rv::iota(0uz, 3uz), rv::iota(0uz, 3uz))) {
            mds_at_ijk[x, y] = std::format("[{}, {}, {}] [{}, {}]", i, j, k, x, y);
        }
    }

    // reading const buff
    const auto& buff_const_ref = buff;
    for (const auto [i, j, k] :
         rv::cartesian_product(rv::iota(0uz, 3uz), rv::iota(0uz, 1uz), rv::iota(0uz, 2uz))) {
        auto mds_at_ijk = buff_const_ref[i, j, k];

        for (const auto [x, y] : rv::cartesian_product(rv::iota(0uz, 3uz), rv::iota(0uz, 3uz))) {
            std::println("[{}, {}, {}] [{}, {}] = {}", i, j, k, x, y, mds_at_ijk[x, y]);
        }
    }
}
