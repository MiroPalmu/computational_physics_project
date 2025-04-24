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

    const auto dfdt = grid.time_derivative();
    std::println("{} = {} = {} = {} = {} = {}",
                 to_str(dfdt.lapse.size()),
                 to_str(dfdt.W.size()),
                 to_str(dfdt.coconf_spatial_metric.size()),
                 to_str(dfdt.K.size()),
                 to_str(dfdt.coconf_A.size()),
                 to_str(dfdt.contraconf_christoffel_trace.size()));
}
