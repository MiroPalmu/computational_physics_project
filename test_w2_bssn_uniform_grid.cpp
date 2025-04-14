#include <array>
#include <print>
#include <vector>

#include "w2_bssn_uniform_grid.cpp"

int
main() {
    auto grid = w2_bssn_uniform_grid({ 2, 3, 4 }, minkowski_spacetime_tag{});
    std::println("sizeof(grid) = {}", sizeof(grid));
    grid.beve_dump();
}
