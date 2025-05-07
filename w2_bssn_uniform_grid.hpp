#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "grid_types.hpp"
#include "tensor_buffer.hpp"

struct minkowski_spacetime_tag {};

/// Represents grid of point on a 3D spatial slice of 4D space time in bssn formalism
/// with W^2 conformal decomposition and harmonic gauge condition with zero shift.
///
/// Points are at coordinates (i, j, k), where {i,j,k} is in [0, N_{i,j,k}).
///
/// From: https://arxiv.org/pdf/gr-qc/0206072
///
/// - Partial derivatives ∂j ˜Γi are computed as finite differences
///   of the independent variables ˜Γi that are evolved using (26).
/// - In all expressions that just require ˜Γi and not its
///   derivative we substitute ˜γjk ˜Γijk (˜γ), that is we do
///   not use the independently evolved variable ˜Γi but
///   recompute ˜Γi according to its definition (14) from
///   the current values of ˜γij .
///
/// In practice we have found that the evolutions are far less
/// stable if either ˜Γi is treated as an independent variable
/// everywhere, or if ˜Γi is recomputed from ˜γij before each
/// time step.
class w2_bssn_uniform_grid {
  public:
    using allocator = sycl::usm_allocator<real, sycl::usm::alloc::shared>;
    using buffer0   = tensor_buffer<0, 3, real, allocator>;
    using buffer1   = tensor_buffer<1, 3, real, allocator>;
    using buffer2   = tensor_buffer<2, 3, real, allocator>;
    using buffer3   = tensor_buffer<3, 3, real, allocator>;

    template<std::size_t rank, typename... T>
    static auto allocate_buffer(T&&... args) {
        return std::allocate_shared<tensor_buffer<rank, 3, real, allocator>>(
            tensor_buffer_queue,
            std::forward<T>(args)...);
    }

    struct time_derivative_type {
        w2_bssn_uniform_grid::buffer0 lapse;
        w2_bssn_uniform_grid::buffer0 W;
        w2_bssn_uniform_grid::buffer2 coconf_metric;
        w2_bssn_uniform_grid::buffer0 K;
        w2_bssn_uniform_grid::buffer2 coconf_A;
        w2_bssn_uniform_grid::buffer1 contraconf_christoffel_trace;

        /// Uninitialized grid with given size.
        [[nodiscard]]
        time_derivative_type(const grid_size);

        /// Applies 6th order Kreiss-Oliger dissipation to derivatives.
        void kreiss_oliger_6th_order(const std::shared_ptr<w2_bssn_uniform_grid>&);
    };

    friend time_derivative_type;

    struct constraints_type {
        w2_bssn_uniform_grid::buffer1 momentum;
        w2_bssn_uniform_grid::buffer0 hamiltonian;

        /// Uninitialized grid with given size.
        [[nodiscard]]
        constraints_type(const grid_size);
    };

    struct pre_calculations_type {
        std::shared_ptr<time_derivative_type> dfdt;
        std::shared_ptr<constraints_type> constraints;
    };

  private:
    grid_size grid_size_;

    buffer0 W_;
    buffer0 lapse_;
    buffer0 K_;

    buffer1 contraconf_christoffel_trace_;

    buffer2 coconf_metric_;
    buffer2 coconf_A_;

  public:
    [[nodiscard]]
    explicit w2_bssn_uniform_grid(const grid_size gs, minkowski_spacetime_tag);

    void beve_dump(const std::filesystem::path& dump_dir_name = "./w2_bssn_uniform_grid_dump");

    [[nodiscard]]
    pre_calculations_type pre_calculations() const;

    [[nodiscard]]
    std::shared_ptr<w2_bssn_uniform_grid> euler_step(const std::shared_ptr<time_derivative_type>&,
                                                     const real) const;

    void enforce_algebraic_constraints();
    void clamp_W(const real);
};

auto
allocate_shared_w2(auto&&... args) {
    return std::allocate_shared<w2_bssn_uniform_grid>(
        w2_bssn_uniform_grid::allocator{ tensor_buffer_queue },
        std::forward<decltype(args)>(args)...);
}
