#include <array>
#include <print>
#include <vector>

#include "w2_bssn_uniform_grid.hpp"

int
main() {
    auto grid = w2_bssn_uniform_grid({ 2, 3, 4 }, minkowski_spacetime_tag{});
    std::println("sizeof(grid) = {}", sizeof(grid));
    grid.beve_dump();

    auto to_str = [](const auto& x) { return std::format("{{ {}, {}, {} }}", x.Nx, x.Ny, x.Nz); };

    const auto pre = grid.pre_calculations();
    std::println("derivative sizes: {} = {} = {} = {} = {} = {}",
                 to_str(pre.dfdt.lapse.size()),
                 to_str(pre.dfdt.W.size()),
                 to_str(pre.dfdt.coconf_metric.size()),
                 to_str(pre.dfdt.K.size()),
                 to_str(pre.dfdt.coconf_A.size()),
                 to_str(pre.dfdt.contraconf_christoffel_trace.size()));

    std::println("constraint sizes: {} = {}",
                 to_str(pre.constraints.momentum.size()),
                 to_str(pre.constraints.hamiltonian.size()));

    static constexpr auto dt   = real{ 0.001 };
    auto first_iter_step = grid.euler_step(pre.dfdt, dt);
    first_iter_step.enforce_algebraic_constraints();
    auto [new_dfdt, _]   = first_iter_step.pre_calculations();

    new_dfdt.kreiss_oliger_6th_order(first_iter_step);
    auto second_iter_step = grid.euler_step(new_dfdt, dt);
    second_iter_step.enforce_algebraic_constraints();
}
