#pragma once
/// @file
#include <algorithm>
#include <array>
#include <format>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stack>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace idg {
namespace str {

/// Checks if \p c is whitespace (std::isspace is not constexpr)
constexpr bool is_whitespace(char8_t const c) {
    // Include your whitespaces here. The example contains the characters
    // documented by https://en.cppreference.com/w/cpp/string/wide/iswspace
    constexpr char8_t matches[] = { ' ', '\n', '\r', '\f', '\v', '\t' };
    return std::any_of(std::begin(matches), std::end(matches), [c](const auto c0) {
        return c == c0;
    });
}

/// Returns the first non-whitespace char or empty optional if there is none
constexpr auto first_non_whitespace_char(const std::u8string_view expr) -> std::optional<char8_t> {
    for (const auto c : expr)
        if (!is_whitespace(c)) return c;
    return {};
}

// Forward decleration for friends of math_string_view
class math_string_view;
class math_string;
constexpr auto split_to_terms(const math_string_view expr) -> std::vector<math_string_view>;
namespace expand_parentheses_impl {
constexpr auto expand_parentheses_vec(const math_string_view str) -> std::vector<std::u8string>;
constexpr auto expand_parentheses(const math_string_view str) -> math_string;
} // namespace expand_parentheses_impl

/// What string_view is for string but for math_string
/*
 * Things that can construct math_string_view:
 *
 * - math_string
 * - selected friend functions for which it makes sense
 *
 * It is concidered bug if any of these constructs a math_string_view
 * for which constructor of math_string would alter the underlying string.
 *
 * This means that one can assume same things as for math_string.
 **/
class math_string_view {
    std::u8string_view sv_;
    [[nodiscard]] explicit constexpr math_string_view(std::u8string_view sv) : sv_{ sv } {}
    // These are given friends so they can construct math_string_view
    // It should
    friend math_string;
    friend constexpr auto split_to_terms(const math_string_view expr)
        -> std::vector<math_string_view>;
    friend constexpr auto
        expand_parentheses_impl::expand_parentheses_vec(const math_string_view str)
            -> std::vector<std::u8string>;

  public:
    [[nodiscard]] constexpr operator std::u8string_view() const { return sv_; }
    [[nodiscard]] constexpr std::u8string_view sv() const { return sv_; }
};

constexpr bool operator==(const math_string_view lhs, const math_string_view rhs) {
    return lhs.sv() == rhs.sv();
}
constexpr bool operator==(const std::u8string_view lhs, const math_string_view rhs) {
    return lhs == rhs.sv();
}
constexpr bool operator==(const math_string_view lhs, const std::u8string_view rhs) {
    return lhs.sv() == rhs;
}

/// std::u8string wrapper which adds structural requirements for the string
/*
 * In each constructor it is checked that:
 *  - Each parentheses is closed
 *  - There is no empty parentheses
 *  - There is no multiple signs (+ or -) in row
 *  - String does not end in sign (+ or -)
 *
 * Also all whitespaces are removed.
 **/
class math_string {
    std::u8string data_;

    /// Checks mathematical requirements of \p str
    /*
     * Checked requirements are:
     *  - Each parentheses is closed
     *  - There is no empty parentheses
     *  - There is no multiple signs (+ or -) in row
     *  - \p does not end in sign (+ or -)
     *
     *  Throws std::logic_error if requirements are not met.
     **/
    constexpr static void check_string(const std::u8string_view str) {
        auto parentheses_level = int{ 0 };
        enum class char_class { sign, open_parentheses, other };
        auto last_non_whitespace_char = char_class::other;

        // (
        auto handle_parentheses_opening = [&] {
            ++parentheses_level;
            last_non_whitespace_char = char_class::open_parentheses;
        };
        // )
        auto handle_parentheses_closing = [&] {
            if (--parentheses_level < 0) throw std::logic_error("Missing matching ( from )!");
            if (last_non_whitespace_char == char_class::open_parentheses)
                throw std::logic_error("Empty parentheses makes no sense!");
            last_non_whitespace_char = char_class::other;
        };
        // + or -
        auto handle_sign = [&] {
            if (last_non_whitespace_char == char_class::sign)
                throw std::logic_error("Multiple signs (+ or -) in row!");
            last_non_whitespace_char = char_class::sign;
        };

        for (const auto c : str) {
            if (is_whitespace(c)) continue;

            switch (c) {
                case u8'(': handle_parentheses_opening(); break;
                case u8')': handle_parentheses_closing(); break;
                case u8'+': [[fallthrough]];
                case u8'-': handle_sign(); break;
                default: last_non_whitespace_char = char_class::other;
            }
        }

        if (parentheses_level > 0) {
            throw std::logic_error("Unmatched ( detected!");
        } else if (parentheses_level < 0) {
            throw std::logic_error("Unmatched ) detected!");
        }

        // Last character
        switch (last_non_whitespace_char) {
            case char_class::sign: throw std::logic_error("Last character is either + or -"); break;
            case char_class::open_parentheses:
                throw std::logic_error("Last character is (!");
                break;
            case char_class::other: break;
            default: std::logic_error("Should not end up here in check_string!");
        }
    };

    /// Copies \p str to \p data_ without whitespace
    constexpr void set_data_without_whitespace(const std::u8string_view str) {
        data_.clear();
        auto is_not_whitespace = [](const auto c) { return not is_whitespace(c); };
        std::ranges::copy_if(str, std::back_inserter(data_), is_not_whitespace);
    }

    friend constexpr auto expand_parentheses_impl::expand_parentheses(const math_string_view str)
        -> math_string;

    /// Used to explicitly mark unchecked constructor
    struct unchecked_tag {};
    /// Unchecked constructor. Should only be called by friend functions.
    /*
     * All friend functions using this should be tested for correctness.
     **/
    [[nodiscard]] constexpr math_string([[maybe_unused]] unchecked_tag,
                                        const std::u8string_view str)
        : data_{ str } {}

  public:
    [[nodiscard]] explicit constexpr math_string(const std::u8string_view str) {
        check_string(str);
        this->set_data_without_whitespace(str);
    }

    [[nodiscard]] constexpr math_string_view msv() const { return math_string_view{ data_ }; };
    [[nodiscard]] constexpr operator math_string_view() const { return math_string_view{ data_ }; }
};

/// Create views to \p mstr by splitting it at signs (+ or -)
/*!
 *  Each term starts with + or - (excpet first might not)
 *  and it's last character is one before next + or -.
 *
 *  Does not split insides of parenhteis.
 *
 *  eg. "A+B(a+b-c)C-DE " -> { "A", "+B(a+b-c)C", "-DE" }
 * */
constexpr auto split_to_terms(const math_string_view mstr) -> std::vector<math_string_view> {
    using namespace std::literals;

    const auto sv_from_indecies = [&](const auto begin_index, const auto one_past_end_index) {
        const auto begin = std::next(mstr.sv().begin(), begin_index);
        const auto end   = std::next(mstr.sv().begin(), one_past_end_index);
        return math_string_view{ std::u8string_view(begin, end) };
    };

    // If first term is minus we have to start iterating at index 1
    // we can assume this is always right due to being impossible to have
    // math_string_view which is just "-".
    if (mstr.sv().empty()) return {};
    const auto first_char_is_minus = mstr.sv().front() == u8'-';

    auto term_begin_index          = 0uz;
    auto terms                     = std::vector<math_string_view>{};
    auto parentheses_nesting_level = 0uz;

    for (const auto [i, c] :
         mstr.sv() | std::views::enumerate | std::views::drop(first_char_is_minus)) {
        if (c == u8'(') {
            parentheses_nesting_level += 1;
            continue;
        } else if (c == u8')') {
            // Due to math_string this can be assumed never to happen
            // if (parentheses_nesting_level == 0) throw std::logic_error{ "Unmatched )" };
            parentheses_nesting_level -= 1;
            continue;
        }

        if (parentheses_nesting_level != 0) {
            // We are inside parentheses
            continue;
        }

        if (i != 0uz and (c == u8'+' or c == u8'-')) {
            terms.push_back(sv_from_indecies(term_begin_index, i));
            term_begin_index = i;
        }
    }

    // Due to math_string this can be assumed never to happen
    // if (parentheses_nesting_level != 0) throw std::logic_error{ "Unmatched (" };

    // last term
    terms.push_back(sv_from_indecies(term_begin_index, mstr.sv().size()));

    return terms;
}

constexpr auto strip_whitespace(const std::u8string_view str) -> std::u8string {
    auto stripped = std::u8string{};
    std::ranges::copy_if(str, std::back_inserter(stripped), [](const auto c) {
        return not is_whitespace(c);
    });
    return stripped;
}

constexpr auto are_same_ignoring_whitespace(const std::u8string_view lhs,
                                            const std::u8string_view rhs) -> bool {
    const auto stripped_lhs = strip_whitespace(lhs);
    const auto stripped_rhs = strip_whitespace(rhs);
    return stripped_lhs == stripped_rhs;
}

/// Like cartesian product but concats strings
constexpr auto cartesian_str_concat(std::span<std::u8string_view const> lhs,
                                    std::span<std::u8string_view const> rhs)
    -> std::vector<std::u8string> {
    auto concats = std::vector<std::u8string>{};

    for (const auto l : lhs) {
        for (const auto r : rhs) { concats.push_back(std::u8string{ l } + std::u8string{ r }); }
    }

    return concats;
}

/// Overload for cartesian_str_concat to work with singular value
constexpr auto cartesian_str_concat(std::u8string_view lhs,
                                    std::span<std::u8string_view const> rhs) {
    const auto span_from_lhs = std::span{ std::addressof(lhs), std::next(std::addressof(lhs)) };
    return cartesian_str_concat(span_from_lhs, rhs);
}

/// Overload for cartesian_str_concat to work with singular value
constexpr auto cartesian_str_concat(std::span<std::u8string_view const> lhs,
                                    std::u8string_view rhs) {
    const auto span_from_rhs = std::span{ std::addressof(rhs), std::next(std::addressof(rhs)) };
    return cartesian_str_concat(lhs, span_from_rhs);
}

/// Overload for cartesian_str_concat to work with singular value
constexpr auto cartesian_str_concat(std::u8string_view lhs, std::u8string_view rhs) {
    const auto span_from_lhs = std::span{ std::addressof(lhs), std::next(std::addressof(lhs)) };
    const auto span_from_rhs = std::span{ std::addressof(rhs), std::next(std::addressof(rhs)) };
    return cartesian_str_concat(span_from_lhs, span_from_rhs);
}

/// Overload for cartesian_str_concat to work with spans of strings
constexpr auto cartesian_str_concat(std::span<std::u8string const> lhs,
                                    std::span<std::u8string const> rhs) {
    auto lhs_as_string_views = std::vector<std::u8string_view>{};
    for (const auto& x : lhs) { lhs_as_string_views.push_back(x); }
    auto rhs_as_string_views = std::vector<std::u8string_view>{};
    for (const auto& x : rhs) { rhs_as_string_views.push_back(x); }

    return cartesian_str_concat(lhs_as_string_views, rhs_as_string_views);
}

/// Overload for cartesian_str_concat to work with spans of strings
constexpr auto cartesian_str_concat(const std::u8string_view lhs,
                                    std::span<std::u8string const> rhs) {
    auto rhs_as_string_views = std::vector<std::u8string_view>{};
    for (const auto& x : rhs) { rhs_as_string_views.push_back(x); }

    return cartesian_str_concat(lhs, rhs_as_string_views);
}

/// Overload for cartesian_str_concat to work with spans of strings
constexpr auto cartesian_str_concat(std::span<std::u8string const> lhs,
                                    const std::u8string_view rhs) {
    auto lhs_as_string_views = std::vector<std::u8string_view>{};
    for (const auto& x : lhs) { lhs_as_string_views.push_back(x); }

    return cartesian_str_concat(lhs_as_string_views, rhs);
}

/// Represents all nested parentheses in math_string_view
class Parentheses {
    using sv_type     = decltype(std::declval<math_string_view>().sv());
    using sv_iterator = sv_type::const_iterator;

    /// Iterator pointing to ( in string
    sv_iterator begin_;
    /// Iterator pointing one past to ) in string
    sv_iterator end_;
    /// Parentheses that are inside of this one
    std::vector<Parentheses> nested_parentheses_ = {};

    using parentheses_ref = std::reference_wrapper<Parentheses>;
    /// std::stack not constexpr :(
    struct constexpr_stack {
        std::vector<parentheses_ref> content = {};
        constexpr std::size_t size() { return content.size(); }
        constexpr parentheses_ref top() { return content.back(); }
        constexpr void pop() { content.pop_back(); }
        constexpr void push(const parentheses_ref x) { content.push_back(x); }
    };

    /// Construct Parentheses from iterators
    /*!
     * We need begin and end not just string view
     * because this might be in half constructed state where
     * we are still looking for the end
     * */
    [[nodiscard]] explicit constexpr Parentheses(const sv_iterator begin,
                                                 const sv_iterator end = nullptr)
        : begin_{ begin },
          end_{ end } {}

  public:
    [[nodiscard]] explicit constexpr Parentheses(const math_string_view mstr)
        : begin_{ mstr.sv().begin() },
          end_{ mstr.sv().end() } {
        // Keep track of in which parenthese loop below is using stack
        auto stack = constexpr_stack{ { *this } };

        for (auto iter = mstr.sv().begin(); iter != mstr.sv().end(); iter = std::next(iter)) {
            if (*iter == u8'(') {
                // Open new parentheses in incomplete state
                stack.top().get().nested_parentheses_.push_back(Parentheses{ iter });
                // Mark new parentheses to be one we are looking ) for
                stack.push(stack.top().get().nested_parentheses_.back());
            } else if (*iter == u8')') {
                // Should not happen due to math_string_view
                // if (stack.size() <= 1uz) throw std::logic_error{ "Unmached )" };

                // Close parentheses
                stack.top().get().end_ = std::next(iter);
                // Mark parentheses done
                stack.pop();
            }
        }

        // Should not happen due to math_string_view
        //if (stack.size() > 1uz) throw std::logic_error{ "Unmached (" };
    }

    /// Deep equality check
    constexpr bool operator==(const Parentheses& rhs) const {
        if (nested_parentheses_.size() != rhs.nested_parentheses_.size()) return false;
        auto are_same = sv_type(begin_, end_) == std::u8string_view(rhs.begin_, rhs.end_);
        if (not are_same) return false;
        using namespace std::literals;
        for (const auto& [lhs_p, rhs_p] :
             std::views::zip(nested_parentheses_, rhs.nested_parentheses_)) {
            are_same = are_same and (lhs_p == rhs_p);
        }
        return are_same;
    }

    [[nodiscard]] constexpr std::span<const Parentheses> nested_parentheses() const {
        return nested_parentheses_;
    }

    [[nodiscard]] constexpr sv_iterator begin() const noexcept { return begin_; }
    [[nodiscard]] constexpr sv_iterator end() const noexcept { return end_; }
    [[nodiscard]] constexpr sv_type sv() const noexcept { return { begin_, end_ }; }
};

/// Expands parentheses in \p str to vector of expanded terms
/*!
 * \p str can be in form:
 *
 * A(B1 + ... + Bn)...(C1 + ... + Cn)D
 *
 * All parentheses are expanded. Appearing terms are returned as a vector.
 *
 * Assume that ...:
 * ... X can be replaced with [+-]X
 * ... + can be replaced with -
 * ... the \p str can have multiple terms in the same form
 *
 * Note that transformation might leave multiple + and - in same term!
 *
 * Does not return std::vector<math_string> because results of this
 * are concatted together using cartesian_str_concat which leads to
 * above, which gets cleaned up in expand_parentheses using filter_plus_and_minus.
 *
 * */
namespace expand_parentheses_impl {
constexpr auto expand_parentheses_vec(const math_string_view mstr) -> std::vector<std::u8string> {
    using namespace std::literals;

    if (mstr.sv().empty()) {
        // Nothing to expand
        return {};
    }

    const auto terms = split_to_terms(mstr);

    if (terms.size() == 1) {
        // ~Base case: processing only one term in form (~ explained below):
        // A(B1 + ... + Bn)...(C1 + ... + Cn)D
        //
        // Do fold type opeartion where binary operation is cartesian string concat

        const auto parentheses_tree = Parentheses{ mstr };

        const auto end_of_initial_lhs = [&] {
            if (parentheses_tree.nested_parentheses().size() == 0) return mstr.sv().end();
            return parentheses_tree.nested_parentheses().front().begin();
        }();

        // This is the base element in fold type operation
        auto lhs = std::vector{ std::u8string(mstr.sv().begin(), end_of_initial_lhs) };

        // ~ because this is base case only when
        // parentheses_tree.nested_parentheses is empy so there is no recursive call
        for (const auto [i, nested_parentheses] :
             parentheses_tree.nested_parentheses() | std::views::enumerate) {
            const auto parentheses_sv = std::u8string_view(std::next(nested_parentheses.begin()),
                                                           std::prev(nested_parentheses.end()));

            // Should not happend due to math_string_view
            // if (are_same_ignoring_whitespace(u8"", parentheses_sv))
            //     throw std::logic_error("Trying to expand empty parentheses!");

            // We know that string coming from Parentheses is correct math_string.
            // See Parentheses::Parentheses(math_string_view) and math_string[_view] documentation.
            // This is why below we are allowed to construct math_string_view using the private
            // constructor (This function is friend of math_string_view).
            //
            // Expanded nested parentheses with recursive call
            const auto expanded = expand_parentheses_vec(math_string_view{ parentheses_sv });

            // This is first binary of two operations
            lhs = cartesian_str_concat(lhs, expanded);

            const auto at_last = i == (std::ssize(parentheses_tree.nested_parentheses()) - 1);
            const auto begin_of_next =
                at_last ? mstr.sv().end() : parentheses_tree.nested_parentheses()[i + 1].begin();

            // This represents stuff after this nested_parentheses but before the next one
            const auto gap_to_next = std::u8string(nested_parentheses.end(), begin_of_next);

            // This is second binary operation
            lhs = cartesian_str_concat(lhs, gap_to_next);
        }

        return lhs;
    }

    // Recursive case: process multiple terms in form:
    // A(B1 + ... + Bn)...(C1 + ... + Cn)D
    //
    // Do fold type operation where binary operation is concatting results of recursive calls
    auto recursive_call = [](const auto term) { return expand_parentheses_vec(term); };
    // Reduntant std::ranges::to is to make this compile on gcc 15 trunk.
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=119282
    return terms | std::views::transform(recursive_call) | std::ranges::to<std::vector>()
           | std::views::join | std::ranges::to<std::vector>();
}

/// Filter + and - and add + if number of - is even and - if odd
/*
 * Should never be called with \p str containing whitespace or parentheses.
 **/
constexpr auto filter_plus_and_minus(const std::u8string_view str) -> std::u8string {
    using namespace std::literals;
    auto minus    = [](const char8_t c) { return c == u8'-'; };
    auto filtered = (std::ranges::count_if(str, minus) % 2 == 0) ? u8"+"s : u8"-"s;

    auto not_plus_or_minus = [](const char8_t c) { return not((c == u8'+') or (c == u8'-')); };
    std::ranges::copy_if(str, std::back_inserter(filtered), not_plus_or_minus);
    return filtered;
}

/// Expands parentheses in \p str
/*!
 * \p str can be in form:
 *
 * A(B1 + ... + Bn)...(C1 + ... + Cn)D
 *
 * All parentheses are expanded. Appearing terms are concated:
 *
 * Assume that ...:
 * ... X can be replaced with [+-]X
 * ... + can be replaced with -
 * ... the \p str can have multiple terms in the same form
 *
 * If transformation results in multiple + and - in same term
 * replace them by + if amount of - is even and - if odd.
 *
 * */
constexpr auto expand_parentheses(const math_string_view str) -> math_string {
    using namespace std::literals;
    auto expanded_str = u8""s;

    const auto terms = expand_parentheses_vec(str);
    for (const auto& term : terms) { expanded_str += filter_plus_and_minus(term); }
    return math_string{ math_string::unchecked_tag{}, expanded_str };
}
} // namespace expand_parentheses_impl
using expand_parentheses_impl::expand_parentheses;

/// Null terminated string which has max length and is structural type
struct fixed_string {
    static constexpr std::size_t max_length   = 100;
    std::array<char8_t, max_length + 1> data_ = {};

    [[nodiscard]] explicit constexpr fixed_string(const std::u8string_view input) {
        if (input.size() > max_length)
            throw std::logic_error("fixed_string max capacity exceeded!");

        std::ranges::copy(input, data_.data());
    }

    template<std::size_t N>
        requires(N <= max_length)
    [[nodiscard]] constexpr fixed_string(const char8_t (&input)[N])
        : fixed_string(static_cast<std::u8string_view>(input)) {}

    [[nodiscard]] constexpr std::u8string_view sv() const {
        return std::u8string_view(data_.begin());
    }
};

} // namespace str
} // namespace idg
