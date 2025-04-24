#include <array>
#include <print>
#include <vector>

#include "w2_bssn_uniform_grid.cpp"

int
main() {
    auto grid = w2_bssn_uniform_grid({ 2, 3, 4 }, minkowski_spacetime_tag{});
    std::println("sizeof(grid) = {}", sizeof(grid));
    grid.beve_dump();

    auto to_str = [](const auto& x) {
        return std::format("{{ {}, {}, {} }}", x.Nx, x.Ny, x.Nz);
    };

    const auto pre = grid.pre_calculations();
    std::println("{} = {} = {} = {} = {} = {}",
                 to_str(pre.dfdt.lapse.size()),
                 to_str(pre.dfdt.W.size()),
                 to_str(pre.dfdt.coconf_metric.size()),
                 to_str(pre.dfdt.K.size()),
                 to_str(pre.dfdt.coconf_A.size()),
                 to_str(pre.dfdt.contraconf_christoffel_trace.size()));
}
