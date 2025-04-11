#include <print>
#include <vector>
#include <array>

#include "w2_bssn_uniform_grid.cpp"

int
main() {
    auto grid = w2_bssn_uniform_grid({ 2, 3, 4 });
    std::println("sizeof(grid) = {}", sizeof(grid));
}
