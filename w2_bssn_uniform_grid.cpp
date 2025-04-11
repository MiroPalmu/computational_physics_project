#include "w2_bssn_uniform_grid.hpp"

[[nodiscard]]
w2_bssn_uniform_grid::w2_bssn_uniform_grid(const grid_size gs)
    : conformal_coefficient_(gs),
      lapse_(gs),
      shift_(gs) {}
