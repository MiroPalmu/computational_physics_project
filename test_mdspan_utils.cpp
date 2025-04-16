#include "ranges.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <print>

#include "idg/sstd.hpp"
#include "mdspan_utils.hpp"

template<typename T>
void
print_mdspan(const T mds) {
    std::println("{{");
    auto counts = std::array<std::size_t, T::rank()>{};

    auto handle_extent_index = [&](const std::size_t I) {
        auto size_uptoa_extent = [&](const std::size_t e) {
            auto c = 1uz;
            for (auto i = 0uz; i <= e; ++i) { c *= mds.extent(i); }
            return c;
        };
        if ((++counts[I] % size_uptoa_extent(I)) == 0uz) { std::println(""); };
    };

    for (const auto idx : mds | idg::sstd::md_indecies) {
        std::print("{:6}, ", std::apply([&](const auto... I) { return mds[I...]; }, idx));
        rn::for_each(rv::iota(0uz) | rv::take(T::rank()), handle_extent_index);
    }
    std::println("}}");
}

int
main() {
    static constexpr auto levi_civita = constant_geometric_mdspan<3, 3>([](const auto idx) {
        const auto [i, j, k] = idx;
        if (i == j or j == k or k == i) { return 0; }
        const auto ij_grows = i < j;
        const auto jk_grows = j < k;
        const auto ki_grows = k < i;
        if (ij_grows + jk_grows + ki_grows == 2) {
            return 1;
        } else {
            return -1;
        }
    });

    print_mdspan(levi_civita);
}
