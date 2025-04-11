#pragma once

#include "ranges.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <experimental/mdspan>

#include "grid_types.hpp"
#include "idg/sstd.hpp"

/// Stores 3D grid of tensors.
template<std::size_t rank, std::size_t D, typename T, typename Allocator>
class tensor_buffer {
    std::size_t Nx_, Ny_, Nz_;
    using vec = std::vector<T, Allocator>;
    std::array<vec, idg::sstd::integer_pow(D, rank)> buffs_;

    /// Access values at specific buffer_index.
    ///
    /// Needs to be templated for const correctness.
    template<typename U>
    struct tensor_buffer_accessor_policy {
        std::size_t buffer_index;

        using element_type     = U;
        using data_handle_type = std::conditional_t<std::is_const_v<U>, vec const*, vec*>;
        using reference        = std::conditional_t<std::is_const_v<U>,
                                                    typename vec::const_reference,
                                                    typename vec::reference>;
        using offset_policy    = tensor_buffer_accessor_policy;

        constexpr reference
        access(this auto&& self, data_handle_type const buff_ptr, const std::size_t i) {
            return buff_ptr[i][self.buffer_index];
        }

        constexpr offset_policy::data_handle_type
        offset(this auto&& self, data_handle_type const buff_ptr, const std::size_t i) {
            return &buff_ptr[i];
        }
    };

    [[nodiscard]]
    constexpr std::size_t total_elements(this auto&& self) {
        return self.Nx_ * self.Ny_ * self.Nz_;
    }

  public:
    [[nodiscard]]
    constexpr explicit tensor_buffer(const std::size_t Nx,
                                     const std::size_t Ny,
                                     const std::size_t Nz)
        : Nx_{ Nx },
          Ny_{ Ny },
          Nz_{ Nz } {
        rn::fill(buffs_, vec(total_elements()));
    }

    [[nodiscard]]
    constexpr explicit tensor_buffer(const grid_size gs)
        : tensor_buffer(gs.Nx, gs.Ny, gs.Nz) {}

    [[nodiscard]]
    constexpr auto
    operator[](this auto&& self, const std::size_t i, const std::size_t j, const std::size_t k) {
        static constexpr auto is_const = std::is_const_v<std::remove_reference_t<decltype(self)>>;
        using element_type             = std::conditional_t<is_const, const T, T>;
        using accessor_policy          = tensor_buffer_accessor_policy<element_type>;
        const auto grid_offset         = (self.Ny_ * self.Nz_ * i) + (self.Nz_ * j) + k;
        return std::mdspan<element_type,
                           idg::sstd::geometric_extents<rank, D>,
                           std::layout_right,
                           accessor_policy>(self.buffs_.data(),
                                            {},
                                            accessor_policy{ .buffer_index = grid_offset });
    }

    [[nodiscard]]
    constexpr auto operator[](this auto&& self, const grid_index idx) {
        return self[idx[0], idx[1], idx[2]];
    }

    template<std::invocable<grid_index, tensor_index<rank>> F>
    constexpr void for_each_index(this auto&& self, F&& f) {
        for (const auto tidx : idg::sstd::geometric_index_space<rank, D>()) {
            for (const auto i : rv::iota(0uz, self.total_elements())) {
                auto idx = grid_index{ (i / (self.Ny_ * self.Nz_)) % self.Nx_,
                                       (i / self.Nz_) % self.Ny_,
                                       i % self.Nz_ };
                std::invoke(std::forward<F>(f), idx, tidx);
            }
        }
    }
};
