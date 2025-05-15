#include "ranges.hpp"

#include <algorithm>
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

#include "env.hpp"
#include "grid_types.hpp"
#include "tensor_buffer.hpp"
#include "w2_bssn_uniform_grid.hpp"

sycl::queue tensor_buffer_queue =
    sycl::queue(sycl::cpu_selector_v, { sycl::property::queue::in_order{} });

int
main(int argc, char** argv) {
    const auto substeps       = read_env("SUBSTEPS", 2uz);
    const auto W_clamp        = read_env("W_CLAMP", real{ 0.0001 });
    const auto dt             = read_env("DT", real{ 0.001 });
    const auto disable_kreiss = read_env("DISABLE_KREISS", false);
    const auto km             = read_env("KM", real{ 0.025 });

    const auto [N, time_steps] = [&] {
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
        return std::tuple{ n, steps };
    }();

    const auto output_interval = rn::max(1uz, static_cast<std::size_t>(time_steps / 100.0));

    const auto output_dir = std::filesystem::path{ "./output" };
    std::filesystem::create_directory(output_dir);
    auto log_file = std::ofstream{ output_dir / "log" };

    // log_file << std::format("grid size: {} x {} x {}\n", N, N, N)
    log_file << std::format("grid size: {} x {} x {}\n", N, 1, 1)
             << std::format("time steps, dt: {}, {}\n", time_steps, dt)
             << std::format("implicit euler substeps: {}\n", substeps)
             << std::format("W clamp: {}\n", W_clamp)
             << std::format("Km (momentum damping coeff): {}\n", km)
             << std::format("Kreiss-Oliger dissipation: {}\n", not disable_kreiss)
             << std::format("output interval: {}\n", output_interval) << std::flush;

    auto step_log_file = std::ofstream{ output_dir / "steps" };

    auto t = real{ 0 };

    // auto base = allocate_shared_w2(grid_size{ N, N, N }, minkowski_spacetime_tag{});
    auto base = allocate_shared_w2(grid_size{ N, 1, 1 }, gauge_wave_spacetime_tag{});

    for (const auto step_ordinal : rv::iota(0uz, time_steps)) {
        const auto start = std::chrono::steady_clock::now();

        step_log_file << std::format("Starting step {}.\n", step_ordinal) << std::flush;

        // take first substep

        base->clamp_W(W_clamp);
        base->enforce_algebraic_constraints();

        auto iter_step = [&] {
            const auto pre1 = base->pre_calculations(km);

            return base->euler_step(pre1.dfdt, dt);
        }();

        const auto make_output = (step_ordinal % output_interval) == 0;
        std::shared_ptr<w2_bssn_uniform_grid::constraints_type> constraints_storage_for_output;

        // other substeps
        for (const auto substep_ordinal : rv::iota(1uz, substeps)) {
            iter_step->clamp_W(W_clamp);
            iter_step->enforce_algebraic_constraints();
            auto pre = iter_step->pre_calculations(km);

            if (not disable_kreiss and substep_ordinal == substeps - 1uz) {
                pre.dfdt->kreiss_oliger_6th_order(base);
            }
            if (make_output) { constraints_storage_for_output = pre.constraints; }
            iter_step = base->euler_step(pre.dfdt, dt);
        }

        base = std::move(iter_step);

        tensor_buffer_queue.wait();

        const auto stop = std::chrono::steady_clock::now();

        step_log_file << std::format("Step done in time: {}\n",
                                     std::chrono::duration<double>(stop - start))
                      << std::flush;

        if (make_output) {
            const auto T = dt * step_ordinal;
            step_log_file << "writing output... ";

            base->append_output(T, *constraints_storage_for_output, output_dir);
        }
    }
}
