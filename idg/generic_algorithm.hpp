#pragma once
/// @file Generic (templated) algorithms
/*
 * Template constraints are offloaded to underling ranges algorithms.
 **/
#include <algorithm>
#include <iterator>
#include <optional>
#include <ranges>
#include <span>

namespace idg {
namespace alg {

/// Returns if \p x appears in \p r once.
template<std::ranges::range R>
[[nodiscard]] constexpr auto appears_once(std::ranges::range_const_reference_t<R> x, R&& r)
    -> bool {
    const auto count = std::ranges::count(std::forward<R>(r), x);
    return count == 1;
}

/// Returns if \p x appears in \p r at most once.
template<std::ranges::range R>
[[nodiscard]] constexpr auto appears_at_most_once(std::ranges::range_const_reference_t<R> x, R&& r)
    -> bool {
    const auto count = std::ranges::count(std::forward<R>(r), x);
    return count <= 1;
}

/// Returns if all elements of \p r appear once in it.
template<std::ranges::forward_range R>
[[nodiscard]] constexpr auto all_appears_once(R&& r) -> bool {
    auto v = std::views::all(std::forward<R&&>(r));
    return std::ranges::all_of(v, [&](std::ranges::range_const_reference_t<R> x) {
        return appears_once(x, v);
    });
}

/// Returns if all elements of \p r appear in \p anohter_r zero or one times.
template<std::ranges::forward_range R1, std::ranges::forward_range R2>
[[nodiscard]] constexpr auto all_appears_at_most_once_in_another(R1&& r, R2&& another_r) -> bool {
    auto another = std::views::all(std::forward<R2>(another_r));
    return std::ranges::all_of(std::forward<R1>(r),
                               [&](std::ranges::range_const_reference_t<R1> x) {
                                   return appears_at_most_once(x, another);
                               });
}

/// Returns if all elements of \p r appear in \p anohter_r exactly once.
template<std::ranges::forward_range R1, std::ranges::forward_range R2>
[[nodiscard]] constexpr auto all_appears_once_in_another(R1&& r, R2&& another_r) -> bool {
    auto another = std::views::all(std::forward<R2>(another_r));
    return std::ranges::all_of(
        std::forward<R1>(r),
        [&](std::ranges::range_const_reference_t<R1> x) { return appears_once(x, another); });
}

/// Return if \p lhs and \p rhs have bijection
/*
 * Notice that if any element appears multiple times in either \p lhs or \p rhs
 * then there is no bijection and and result is false.
 **/
template<std::ranges::forward_range R1, std::ranges::forward_range R2>
[[nodiscard]] constexpr auto have_bijection(R1&& lhs, R2&& rhs) -> bool {
    auto lhs_view = std::views::all(std::forward<R1>(lhs));
    auto rhs_view = std::views::all(std::forward<R2>(rhs));
    return all_appears_once_in_another(lhs_view, rhs_view)
           && all_appears_once_in_another(rhs_view, lhs_view);
}

template<std::ranges::input_range R>
[[nodiscard]] constexpr std::optional<std::size_t>
    argfind(R&& r, std::ranges::range_const_reference_t<R> x) {
    namespace rn  = std::ranges;
    const auto b  = rn::begin(r);
    const auto e  = rn::end(r);
    const auto ix = rn::find(r, x);
    if (ix == e) {
        return std::nullopt;
    } else {
        return static_cast<std::size_t>(rn::distance(b, ix));
    }
}

} // namespace alg
} // namespace idg
