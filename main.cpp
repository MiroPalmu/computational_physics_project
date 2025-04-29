#include "ranges.hpp"

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <ranges>

#include "grid_types.hpp"
#include "w2_bssn_uniform_grid.hpp"

static constexpr auto N          = 20uz;
static constexpr auto time_steps = 10uz;
// Deduced from NR101 which has simulation width of 1 and time step 0.001.
static constexpr auto dt       = real{ 0.001 * N };
static constexpr auto substeps = 2uz;
static constexpr auto W_clamp  = real{ 0.0001 };

int
main() {
    const auto output_dir = std::filesystem::path{ "./output" };
    std::filesystem::create_directory(output_dir);
    auto log_file = std::ofstream{ output_dir / "log" };
    log_file << std::format("grid size: {} x {} x {}\n", N, N, N)
             << std::format("time steps, dt: {}, {}\n", time_steps, dt)
             << std::format("implicit euler substeps: {}\n", substeps)
             << std::format("W clamp: {}\n", W_clamp) << std::flush;

    auto step_log_file = std::ofstream{ output_dir / "steps" };

    auto t = real{ 0 };

    auto base = w2_bssn_uniform_grid({ N, N, N }, minkowski_spacetime_tag{});

    for (const auto step_ordinal : rv::iota(0uz, time_steps)) {
        const auto start = std::chrono::steady_clock::now();

        step_log_file << std::format("Starting step {}.\n", step_ordinal);

        // take first substep

        base.clamp_W(W_clamp);
        base.enforce_algebraic_constraints();
        const auto [dfdt1, _] = base.pre_calculations();

        auto iter_step = base.euler_step(dfdt1, dt);

        // other substeps
        for (const auto substep_ordinal : rv::iota(1uz, substeps)) {
            iter_step.clamp_W(W_clamp);
            iter_step.enforce_algebraic_constraints();
            auto [dfdt, _] = iter_step.pre_calculations();

            if (substep_ordinal == substeps - 1uz) { dfdt.kreiss_oliger_6th_order(base); }

            iter_step = base.euler_step(dfdt, dt);
        }

        base = std::move(iter_step);

        const auto stop = std::chrono::steady_clock::now();

        step_log_file << std::format("Step done in time: {}\n",
                                     std::chrono::duration<double>(stop - start))
                      << std::flush;
    }
}
