#include "ranges.hpp"

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <ranges>
#include <sstream>
#include <string_view>
#include <tuple>

#include <sycl/sycl.hpp>

#include "grid_types.hpp"
#include "tensor_buffer.hpp"
#include "w2_bssn_uniform_grid.hpp"

static constexpr auto substeps = 2uz;
static constexpr auto W_clamp  = real{ 0.0001 };

sycl::queue tensor_buffer_queue =
    sycl::queue(sycl::cpu_selector_v, { sycl::property::queue::in_order{} });

int
main(int argc, char** argv) {
    const auto [N, time_steps, dt] = [&] {
        if (argc != 3) {
            std::println("Wrong amount of arguments:!");
            std::println("usage: main N time_steps");
            std::exit(1);
        }

        std::stringstream ss;
        ss << argv[1] << " " << argv[2];

        std::size_t n, steps;
        ss >> n >> steps;
        // Deduced from NR101 which has simulation width of 1 and time step 0.001.
        return std::tuple{ n, steps, 0.001 * n };
    }();

    const auto output_dir = std::filesystem::path{ "./output" };
    std::filesystem::create_directory(output_dir);
    auto log_file = std::ofstream{ output_dir / "log" };
    log_file << std::format("grid size: {} x {} x {}\n", N, N, N)
             << std::format("time steps, dt: {}, {}\n", time_steps, dt)
             << std::format("implicit euler substeps: {}\n", substeps)
             << std::format("W clamp: {}\n", W_clamp) << std::flush;

    auto step_log_file = std::ofstream{ output_dir / "steps" };

    auto t = real{ 0 };

    auto base = allocate_shared_w2(grid_size{ N, N, N }, minkowski_spacetime_tag{});

    for (const auto step_ordinal : rv::iota(0uz, time_steps)) {
        const auto start = std::chrono::steady_clock::now();

        step_log_file << std::format("Starting step {}.\n", step_ordinal) << std::flush;

        // take first substep

        base->clamp_W(W_clamp);
        base->enforce_algebraic_constraints();

        auto iter_step = [&] {
            const auto pre1 = base->pre_calculations();

            return base->euler_step(pre1.dfdt, dt);
        }();

        // other substeps
        for (const auto substep_ordinal : rv::iota(1uz, substeps)) {
            iter_step->clamp_W(W_clamp);
            iter_step->enforce_algebraic_constraints();
            auto pre = iter_step->pre_calculations();

            if (substep_ordinal == substeps - 1uz) { pre.dfdt->kreiss_oliger_6th_order(base); }

            iter_step = base->euler_step(pre.dfdt, dt);
        }

        base = std::move(iter_step);

        tensor_buffer_queue.wait();

        const auto stop = std::chrono::steady_clock::now();

        step_log_file << std::format("Step done in time: {}\n",
                                     std::chrono::duration<double>(stop - start))
                      << std::flush;
    }
}
