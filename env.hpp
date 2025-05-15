#pragma once

#include <cstdlib>
#include <sstream>
#include <string>
#include <type_traits>

auto
read_env(const std::string& var, const auto default_val) {
    if (const char* env_ptr = std::getenv(var.c_str())) {
        std::remove_cvref_t<decltype(default_val)> val;
        auto ss = std::stringstream{ env_ptr };
        ss >> val;
        return val;
    }

    return default_val;
}
