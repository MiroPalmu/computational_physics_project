#pragma once

#include <array>

using real = float;

struct grid_size {
    std::size_t Nx, Ny, Nz;
};

using grid_index = std::array<std::size_t, 3>;

template <std::size_t rank>
using tensor_index = std::array<std::size_t, rank>;
