#include <print>
#include <vector>

#include <experimental/mdspan>

#include "sycl/sycl.hpp"

#include "idg/einsum.hpp"
#include "idg/sstd.hpp"


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


    auto buff = std::vector<double>(4, 2);
    auto mds = idg::sstd::geometric_mdspan<double, 1, 4>(buff.data());

    double out_value;
    auto out_mds = idg::sstd::geometric_mdspan<double, 0, 4>(&out_value);

    using namespace idg::literals;
    u8"i,i"_einsum(out_mds, mds, mds);
    std::println("{}", out_value);
}
