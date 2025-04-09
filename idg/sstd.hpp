#pragma once
/* @file Supplement Standard library (sstd)
 *
 * This header implements some cpp standard features
 * that are not yet implemented on compilers.
 **/
#include <algorithm>
#include <array>
#include <functional>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <experimental/mdspan>

#include "idg/generic_algorithm.hpp"

namespace idg {
namespace sstd {

// clang-format off
/// Converts array<T, N> to tuple<T, ..., T> with values of array
template<typename T, std::size_t N>
// libstdc++ ok but libc++ not.
// #if __cpp_lib_tuple_like >= 202207L
// [[depricated("use tuple's tuple-like constructor")]]
// #endif
constexpr auto array_as_tuple(const std::array<T, N>& x) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return std::make_tuple(x[I]...);
    }(std::make_index_sequence<N>{});
}
// clang-format on

template<typename T>
struct is_mdspan : std::false_type {};

template<typename T, typename E, typename LP, typename AP>
struct is_mdspan<std::mdspan<T, E, LP, AP>> : std::true_type {};

template<typename T>
static constexpr bool is_mdspan_v = is_mdspan<T>::value;

/// Range adaptor to iterate over mdspan indeceis in arbitrary order
struct md_indecies_type : std::ranges::range_adaptor_closure<md_indecies_type> {
    template<typename T>
        requires is_mdspan_v<T>
    [[nodiscard]] constexpr auto operator()(const T mds) const {
        using index_type = decltype(mds)::index_type;
        auto indecies =
            std::array<decltype(std::views::iota(index_type{ 0 }, mds.extent(0))), mds.rank()>{};

        for (const auto l : std::views::iota(index_type{ 0 }, mds.rank())) {
            indecies[l] = std::views::iota(index_type{ 0 }, mds.extent(l));
        }
        // Here one could use apply tuple-like version and not convert indecies to tuple
        // but gcc 13.2 does not support it yet
        return std::apply(std::views::cartesian_product, array_as_tuple(indecies))
               | std::views::transform([](const auto& t) {
                     return std::apply([](const auto... i) { return std::array{ i... }; }, t);
                 });
    }
};
inline constexpr auto md_indecies = md_indecies_type{};

template<typename T>
class constexpr_set {
    std::vector<T> data_;

  public:
    using value_type = typename decltype(data_)::value_type;

    [[nodiscard]] constexpr bool contains(const value_type& x) const {
        return std::ranges::find(data_, x) != std::ranges::end(data_);
    }

    /// Insert element if not already in and return if insert happened
    [[nodiscard]] constexpr bool successfully_insert(const value_type& x) {
        if (contains(x)) return false;
        data_.push_back(x);
        return true;
    }

    /// There is one to one match of elements of this set to \p rhs
    [[nodiscard]] constexpr bool operator==(const constexpr_set<value_type>& rhs) const {
        return idg::alg::all_appears_once_in_another(data_, rhs.data_)
               and idg::alg::all_appears_once_in_another(rhs.data_, data_);
    }

    /// Give view of the underlying data in unspecified order.
    [[nodiscard]] constexpr auto get_data() const -> std::span<value_type const> { return data_; }
};

namespace flags {
template<class T>
    requires std::is_enum_v<T> and std::unsigned_integral<std::underlying_type_t<T>>
[[nodiscard]] constexpr T operator|(const T lhs, const T rhs) noexcept {
    return static_cast<T>(static_cast<std::underlying_type<T>::type>(lhs)
                          | static_cast<std::underlying_type<T>::type>(rhs));
}

template<class T>
    requires std::is_enum_v<T> and std::unsigned_integral<std::underlying_type_t<T>>
[[nodiscard]] constexpr T operator|(const std::underlying_type_t<T> lhs, const T rhs) noexcept {
    return static_cast<T>(static_cast<std::underlying_type<T>::type>(lhs)
                          | static_cast<std::underlying_type<T>::type>(rhs));
}

/// flags represents flags which are composable with operator|
/*
 * Flags are represented as enums which have usigned integer as underlying type.
 * The enum values have to be given explicitly as zero or any positive power of two.
 *
 * This allows to use enums like bit flags and use helper operator|s
 * implemented in idg::sstd::flags namespace to combine them.
 *
 * Example usage:
 * ```cpp
 * using namespace idg::sstd::flags;
 * enum class flags uint8_t { a = 0, b = 1, c = 2, d = 4, ... };
 *
 * auto foo = flags_t<flags>(flags::a);
 * foo.add(flags::b);
 * foo.add(flags::a | flags::b);
 * foo.contains(flags::a | flags::b); // == true
 * foo.remove(flags::a);
 * foo.contains(flags::a | flags::b); // == false
 * foo.contains(flags::b);            // == true
 * ```
 **/
template<typename T>
    requires std::is_enum_v<T> and std::unsigned_integral<std::underlying_type_t<T>>
class flags_t {
  public:
    using UT = std::underlying_type_t<T>;

  private:
    UT current_flags;

  public:
    constexpr bool operator==(const flags_t&) const noexcept = default;

    constexpr void remove(const flags_t flags) noexcept { remove(flags.current_flags); }
    constexpr void add(const flags_t flags) noexcept { add(flags.current_flags); }
    [[nodiscard]] constexpr bool contains(const flags_t flags) const noexcept {
        return contains(flags.current_flags);
    }

    // Using underlying type
    constexpr flags_t(const UT initial_flag) noexcept : current_flags{ initial_flag } {}
    constexpr void remove(const UT flag) noexcept { current_flags &= ~flag; }
    constexpr void add(const UT flag) noexcept { current_flags |= flag; }
    [[nodiscard]] constexpr bool contains(const UT flag) const noexcept {
        return not static_cast<bool>(flag & ~current_flags);
    }

    friend constexpr bool operator==(const flags_t lhs, const UT rhs) noexcept {
        return lhs.current_flags == rhs;
    };
    friend constexpr bool operator==(const UT lhs, const flags_t rhs) noexcept {
        return lhs == rhs.current_flags;
    };

    // Using enum
    constexpr flags_t(const T initial_flags) : flags_t(static_cast<UT>(initial_flags)) {}
    constexpr void remove(const T flags) noexcept { remove(static_cast<UT>(flags)); }
    constexpr void add(const T flags) noexcept { add(static_cast<UT>(flags)); }
    [[nodiscard]] constexpr bool contains(const T flags) const noexcept {
        return contains(static_cast<UT>(flags));
    }

    friend constexpr bool operator==(const flags_t lhs, const T rhs) noexcept {
        return lhs.current_flags == static_cast<UT>(rhs);
    };
    friend constexpr bool operator==(const T lhs, const flags_t rhs) noexcept {
        return static_cast<UT>(lhs) == rhs.current_flags;
    };
};

} // namespace flags

/// Compare if spans are elementwise equal
template<std::equality_comparable T>
constexpr bool compare_spans(const std::span<const T> lhs, const std::span<const T> rhs) {
    auto are_same = lhs.size() == rhs.size();
    if (not are_same) return false;
    for (const auto& [l, r] : std::views::zip(lhs, rhs)) are_same = are_same and (l == r);
    return are_same;
}

template<std::integral T>
[[nodiscard]] constexpr T integer_pow(const T base, const T exponent) {
    if (exponent < T{ 0 }) {
        throw std::logic_error{ "Integers can not be raised to negative power." };
    }

    auto result = T{ 1 };
    for (auto _ : std::views::iota(T{ 0 }, exponent)) { result *= base; }
    return result;
}

template<std::size_t rank, std::size_t dim>
using geometric_extents = decltype(std::invoke(
    []<std::size_t... I>(std::index_sequence<I...>) {
        return std::extents<std::size_t, (0 * I + dim)...>{};
    },
    std::make_index_sequence<rank>()));

template<typename T,
         std::size_t rank,
         std::size_t dim,
         typename LayoutPolicy   = std::layout_right,
         typename AccessorPolicy = std::default_accessor<T>>
using geometric_mdspan = std::mdspan<T, geometric_extents<rank, dim>, LayoutPolicy, AccessorPolicy>;

template<std::size_t rank, std::size_t D>
[[nodiscard]] consteval auto geometric_index_space() {
    namespace rv = std::ranges::views;

    return std::invoke(
        []<std::size_t... I>(std::index_sequence<I...>) {
            return rv::cartesian_product(rv::iota(I * 0uz, D)...)
                   | std::views::transform([](const auto& t) {
                         return std::apply(
                             [](const auto... i) { return std::array<std::size_t, rank>{ i... }; },
                             t);
                     });
        },
        std::make_index_sequence<rank>());
};

/// Natural means that it works only with natural coefficients and exponents.
class natural_polynomial {
    /// i:th coeff corresponds to x^i
    ///
    /// It is assumed that member functions keep this exactly as long
    /// as the largest x^i requires.
    std::vector<std::size_t> coeffs_{};

    [[nodiscard]] constexpr natural_polynomial(std::vector<std::size_t> coeffs) {
        coeffs_ = std::move(coeffs);
    }

  public:
    [[nodiscard]] constexpr natural_polynomial() = default;

    [[nodiscard]] explicit constexpr natural_polynomial(const std::size_t exponent) {
        coeffs_.resize(exponent + 1uz);
        std::ranges::fill(coeffs_, 0uz);
        coeffs_[exponent] = 1uz;
    }

    [[nodiscard]] constexpr std::size_t evalf(this auto&& self, const std::size_t x) {
        namespace rv = std::views;
        const auto vals =
            self.coeffs_ | rv::enumerate | rv::transform([x](const auto t) {
                return std::get<1>(t) * integer_pow(x, static_cast<std::size_t>(std::get<0>(t)));
            });
        return std::reduce(vals.begin(), vals.end(), 0uz);
    }

    [[nodiscard]] friend constexpr natural_polynomial operator+(const natural_polynomial& lhs,
                                                                const natural_polynomial& rhs) {
        auto new_coeffs =
            std::vector<std::size_t>(std::ranges::max(lhs.coeffs_.size(), rhs.coeffs_.size()), 0uz);

        for (const auto [i, c] : lhs.coeffs_ | std::views::enumerate) { new_coeffs[i] += c; }
        for (const auto [i, c] : rhs.coeffs_ | std::views::enumerate) { new_coeffs[i] += c; }

        return natural_polynomial(std::move(new_coeffs));
    }

    [[nodiscard]] friend constexpr natural_polynomial operator*(const std::size_t c,
                                                                const natural_polynomial& rhs) {
        if (c == 0uz) { return natural_polynomial{}; }
        return natural_polynomial(rhs.coeffs_
                                  | std::views::transform([c](const auto x) { return c * x; })
                                  | std::ranges::to<std::vector>());
    }

    [[nodiscard]] friend constexpr natural_polynomial operator*(const natural_polynomial& lhs,
                                                                const natural_polynomial& rhs) {
        if (std::ranges::empty(lhs.coeffs_) or std::ranges::empty(rhs.coeffs_)) { return {}; }

        auto new_coeffs =
            std::vector<std::size_t>(1uz + (lhs.coeffs_.size() - 1uz) + (rhs.coeffs_.size() - 1uz));

        auto cart = std::views::cartesian_product(lhs.coeffs_ | std::views::enumerate,
                                                  rhs.coeffs_ | std::views::enumerate);

        for (const auto [l, r] : cart) {
            const auto lhs_c   = std::get<1>(l);
            const auto rhs_c   = std::get<1>(r);
            const auto lhs_exp = std::get<0>(l);
            const auto rhs_exp = std::get<0>(r);

            new_coeffs[lhs_exp + rhs_exp] = lhs_c * rhs_c;
        }

        return natural_polynomial(std::move(new_coeffs));
    }

    [[nodiscard]] friend constexpr bool operator==(const natural_polynomial&,
                                                   const natural_polynomial&) = default;
};

} // namespace sstd
} // namespace idg
