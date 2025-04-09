#include <print>

#include "sycl/sycl.hpp"

int
main() {
    // Loop through available platforms
    for (auto const& platform : sycl::platform::get_platforms()) {
        std::println("Found platform: {}", platform.get_info<sycl::info::platform::name>());
        // Loop through available devices in this platform
        for (auto const& device : platform.get_devices()) {
            std::println("With device: ", device.get_info<sycl::info::device::name>());
        }
        std::println("");
    }
}
