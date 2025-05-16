#pragma once

#include "ranges.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <type_traits>
#include <utility>
#include <vector>

#include <experimental/mdspan>

#include <sycl/sycl.hpp>

#include "grid_types.hpp"
#include "idg/sstd.hpp"

extern sycl::queue tensor_buffer_queue;

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
          Nz_{ Nz },
          buffs_{ [&]<std::size_t... I>(std::index_sequence<I...>) {
              if constexpr (std::same_as<Allocator,
                                         sycl::usm_allocator<real, sycl::usm::alloc::shared>>) {
                  return std::array{ (std::ignore = I,
                                      vec(total_elements(),
                                          sycl::usm_allocator<real, sycl::usm::alloc::shared>(
                                              tensor_buffer_queue)))... };
              } else {
                  return std::array{ (std::ignore = I, vec(total_elements()))... };
              }
          }(std::make_index_sequence<std::tuple_size<decltype(buffs_)>{}>()) } {}

    [[nodiscard]]
    constexpr explicit tensor_buffer(const grid_size gs)
        : tensor_buffer(gs.Nx, gs.Ny, gs.Nz) {}

    [[nodiscard]]
    constexpr grid_size size(this auto&& self) {
        return { self.Nx_, self.Ny_, self.Nz_ };
    }

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

    template<typename F>
        requires std::invocable<F, grid_index, tensor_index<rank>> or std::invocable<F, grid_index>
    constexpr void for_each_index(this auto&& self, F&& f) {
        if constexpr (std::invocable<F, grid_index>) {
            tensor_buffer_queue.parallel_for(sycl::range{ self.Nx_, self.Ny_, self.Nz_ },
                                             [f = std::forward<F>(f)](sycl::id<3> idx) {
                                                 f(grid_index{ idx[0], idx[1], idx[2] });
                                             });
        } else {
            for (const auto tidx : idg::sstd::geometric_index_space<rank, D>()) {
                tensor_buffer_queue.parallel_for(
                    sycl::range{ self.Nx_, self.Ny_, self.Nz_ },
                    [=](sycl::id<3> idx) { f(grid_index{ idx[0], idx[1], idx[2] }, tidx); });
            }
        }
    }
};


/// Syntax sugar for following:
///
/// [foo_ptr = foo_ptr.get(), bar_ptr = bar_ptr.get()]
///
/// to
///
/// [SPTR(foo_ptr, bar_ptr)]

#define GET_ARG(arg) arg = arg.get()

#define FOR_EACH_1(action, x) action(x)
#define FOR_EACH_2(action, x, ...) action(x), FOR_EACH_1(action, __VA_ARGS__)
#define FOR_EACH_3(action, x, ...) action(x), FOR_EACH_2(action, __VA_ARGS__)
#define FOR_EACH_4(action, x, ...) action(x), FOR_EACH_3(action, __VA_ARGS__)
#define FOR_EACH_5(action, x, ...) action(x), FOR_EACH_4(action, __VA_ARGS__)
#define FOR_EACH_6(action, x, ...) action(x), FOR_EACH_5(action, __VA_ARGS__)
#define FOR_EACH_7(action, x, ...) action(x), FOR_EACH_6(action, __VA_ARGS__)
#define FOR_EACH_8(action, x, ...) action(x), FOR_EACH_7(action, __VA_ARGS__)
#define FOR_EACH_9(action, x, ...) action(x), FOR_EACH_8(action, __VA_ARGS__)
#define FOR_EACH_10(action, x, ...) action(x), FOR_EACH_9(action, __VA_ARGS__)

// Macro to count number of args (up to 10)
#define GET_MACRO(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,NAME,...) NAME
#define FOR_EACH(action, ...) \
  GET_MACRO(__VA_ARGS__, \
            FOR_EACH_10, FOR_EACH_9, FOR_EACH_8, FOR_EACH_7, \
            FOR_EACH_6, FOR_EACH_5, FOR_EACH_4, FOR_EACH_3, \
            FOR_EACH_2, FOR_EACH_1)(action, __VA_ARGS__)

#define SPTR(...) FOR_EACH(GET_ARG, __VA_ARGS__)
