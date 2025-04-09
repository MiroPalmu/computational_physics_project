#pragma once

#include <algorithm>
#include <iterator>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include "idg/generic_algorithm.hpp"
#include "idg/sstd.hpp"

namespace idg {

namespace rn = std::ranges;
namespace rv = std::ranges::views;

class connected_tensor_network;

class tensor_network {
  public:
    struct node_id {
        std::size_t id;

        [[nodiscard]] friend constexpr bool operator==(const node_id&, const node_id&) = default;
    };

    struct index_location {
        node_id id;
        std::size_t index;

        [[nodiscard]] friend constexpr bool operator==(const index_location&,
                                                       const index_location&) = default;
    };

    struct node {
        node_id id;
        std::size_t rank;

        [[nodiscard]] friend constexpr bool operator==(const node&, const node&) = default;
    };

    struct edge {
        index_location left;
        index_location right;

        /// It does not matter in which order left and right are.
        [[nodiscard]] friend constexpr bool operator==(const edge& lhs, const edge& rhs) {
            return (lhs.left == rhs.left and lhs.right == rhs.right)
                   or (lhs.left == rhs.right and lhs.right == rhs.left);
        };
    };

  protected:
    std::vector<node> nodes_{};
    std::size_t next_id_{ 0uz };
    std::vector<edge> edges_{};

    [[nodiscard]] constexpr std::size_t rank_wout_reductions(this auto&& self) {
        auto node_ranks = self.nodes_ | rv::transform([](const node& n) { return n.rank; });
        return std::reduce(node_ranks.begin(), node_ranks.end(), 0uz);
    }

    /// Caller has to make sure other does not share node ids with self.
    constexpr void connect(this tensor_network& self, const tensor_network& other, const edge e) {
        // No std::vector::append_range in gcc 14 :(
        self.nodes_.insert(self.nodes_.end(), other.nodes_.begin(), other.nodes_.end());
        self.edges_.insert(self.edges_.end(), other.edges_.begin(), other.edges_.end());
        self.edges_.push_back(e);
        self.next_id_ = rn::max(self.next_id_, other.next_id_) + 1;
    }

  public:
    [[nodiscard]] constexpr node_id add_node(this tensor_network& self, const std::size_t rank) {
        self.nodes_.push_back({ .id = { self.next_id_ }, .rank = rank });
        return { self.next_id_++ };
    }

    [[nodiscard]] constexpr std::size_t size(this auto&& self) { return rn::size(self.nodes_); }

    [[nodiscard]] constexpr bool contains(this auto&& self, const node_id id) {
        return rn::contains(self.nodes_, id, &node::id);
    }

    [[nodiscard]] constexpr std::size_t rank(this auto&& self) {
        const auto N = self.rank_wout_reductions();
        return N - 2uz * rn::size(self.edges_);
    }

    [[nodiscard]] constexpr std::vector<connected_tensor_network>
        connected_components(this auto&& self);

    constexpr void
        add_edge(this tensor_network& self, const index_location a, const index_location b) {
        if (a == b) {
            throw std::logic_error{ "Can not add edge from a index to the same index." };
        }

        const auto node_a_ptr = rn::find(self.nodes_, a.id, &node::id);
        const auto node_b_ptr = rn::find(self.nodes_, b.id, &node::id);

        if (node_a_ptr == rn::end(self.nodes_) or node_b_ptr == rn::end(self.nodes_)) {
            throw std::logic_error{ "Trying to add edge to non-existing node." };
        }

        if (node_a_ptr->rank <= a.index or node_b_ptr->rank <= b.index) {
            throw std::logic_error{
                "Trying to add edge to non-existing index (index >= node rank)."
            };
        }

        if (self.edges_.end() != rn::find_if(self.edges_, [&](const auto& e) {
                return e.left == a or e.left == b or e.right == a or e.right == b;
            })) {
            throw std::logic_error{ "Trying to add second edge to the same index." };
        }

        self.edges_.push_back({ a, b });
    }

    [[nodiscard]] constexpr rn::view auto view_edges(this const tensor_network& self) {
        return rv::all(self.edges_) | rv::as_const;
    }

    [[nodiscard]] constexpr rn::view auto view_nodes(this const tensor_network& self) {
        return rv::all(self.nodes_) | rv::as_const;
    }
};

class connected_tensor_network;

class pairwise_contraction_type {
    friend class connected_tensor_network;

    tensor_network::node lhs_, rhs_;
    std::optional<tensor_network::node> out_{};
    std::vector<tensor_network::edge> edges_;

    [[nodiscard]] constexpr pairwise_contraction_type(const tensor_network::node lhs,
                                                      const tensor_network::node rhs,
                                                      rn::input_range auto&& reduction_edges)
        : lhs_{ lhs },
          rhs_{ rhs },
          edges_(reduction_edges.begin(), reduction_edges.end()) {
        if (rn::any_of(edges_, [&](const tensor_network::edge& e) {
                return (e.left.id != lhs.id and e.left.id != rhs.id)
                       or (e.right.id != lhs.id and e.right.id != rhs.id);
            })) {
            throw std::logic_error{
                "Can not have reduction edge which does not end at lhs nor rhs."
            };
        }
    }

  public:
    [[nodiscard]] constexpr std::size_t cost(this auto&& self, const std::size_t D) {
        return sstd::integer_pow(D, self.lhs_.rank + self.rhs_.rank - rn::size(self.edges_));
    }

    [[nodiscard]] constexpr tensor_network::node_id lhs_id(this auto&& self) {
        return self.lhs_.id;
    }

    [[nodiscard]] constexpr tensor_network::node_id rhs_id(this auto&& self) {
        return self.rhs_.id;
    }

    [[nodiscard]] constexpr tensor_network::node_id out_id(this auto&& self) {
        return self.out_.value().id;
    }

    constexpr void store_out(this pairwise_contraction_type& self, const tensor_network::node out) {
        self.out_ = out;
    }

    [[nodiscard]] constexpr std::size_t out_rank(this auto&& self) {
        return self.out_.value().rank;
    }

    [[nodiscard]] constexpr std::pair<std::u8string, std::u8string> index_labels(this auto&& self) {
        auto lhs_str = std::u8string(self.lhs_.rank, u8' ');
        auto rhs_str = std::u8string(self.rhs_.rank, u8' ');

        auto increment_char8 = [](const char8_t c8) {
            const auto c = static_cast<char>(c8);
            return static_cast<char8_t>(c + 1);
        };

        lhs_str[0] = u8'a';
        for (const auto i : rv::iota(1uz, lhs_str.size())) {
            lhs_str[i] = increment_char8(lhs_str[i - 1uz]);
        }

        rhs_str[0] = increment_char8(lhs_str.back());
        for (const auto i : rv::iota(1uz, rhs_str.size())) {
            rhs_str[i] = increment_char8(rhs_str[i - 1uz]);
        }

        for (const auto& e : self.edges_) {
            char lhs_index_label;

            if (e.left.id == self.lhs_.id) {
                lhs_index_label = lhs_str[e.left.index];
            } else {
                lhs_index_label = rhs_str[e.left.index];
            }

            if (e.right.id == self.rhs_.id) {
                rhs_str[e.right.index] = lhs_index_label;
            } else {
                lhs_str[e.right.index] = lhs_index_label;
            }
        }

        return { std::move(lhs_str), std::move(rhs_str) };
    }
};

template<rn::input_range R>
[[nodiscard]] constexpr std::size_t contraction_cost(R&& pcs, const std::size_t D) {
    auto pcs_view = rv::all(std::forward<R>(pcs))
                    | rv::transform([&](const auto& contraction) { return contraction.cost(D); });

    return std::reduce(pcs_view.begin(), pcs_view.end());
}

class connected_tensor_network : public tensor_network {
    friend class tensor_network;

  private:
    using tensor_network::add_node;
    using tensor_network::add_edge;

    /// Each element corresponds to all edges contracted in pairwise contraction.
    ///
    /// There is a group for all node pairs which have connecting edge,
    /// so edges from node to itself, might be contained in multiple groups.
    [[nodiscard]] constexpr auto group_edges_pairwise(this auto&& self) {
        if (self.size() <= 1uz) {
            throw std::logic_error{ "Grouping does not make sense for single node." };
        }

        auto groups     = std::vector<std::vector<edge>>{};
        auto node_pairs = std::vector<std::pair<node, node>>{};

        auto edge_part_of_node_pair = [&](const edge& e, const node_id node1, const node_id node2) {
            return (e.left.id == node1 or e.left.id == node2)
                   and (e.right.id == node1 or e.right.id == node2);
        };

        auto internode_edge = [](const edge& e) { return e.left.id != e.right.id; };

        for (const auto [i, n1] : self.view_nodes() | rv::enumerate) {
            for (const auto n2 : self.view_nodes() | rv::drop(i + 1)) {
                groups.push_back({});
                for (const auto e : self.view_edges()) {
                    if (edge_part_of_node_pair(e, n1.id, n2.id)) { groups.back().push_back(e); }
                }
                if (rn::none_of(groups.back(), internode_edge)) {
                    // group not accepted
                    groups.resize(groups.size() - 1uz);
                } else {
                    // group accepted
                    node_pairs.push_back({ n1, n2 });
                }
            }
        }

        if (rn::empty(groups)) { throw std::logic_error{ "There should be at least one group." }; }

        if (rn::size(groups) != rn::size(node_pairs)) {
            throw std::logic_error{ "Different amount of groups and node pairs." };
        }
        return std::pair{ node_pairs, groups };
    }

  public:
    [[nodiscard]] constexpr node_id pairwise_contraction(this connected_tensor_network& self,
                                                         const node_id lhs,
                                                         const node_id rhs) {
        if (lhs == rhs) {
            throw std::logic_error{ "Can not pairwise contract a node with itself." };
        }
        const auto lhs_node_ptr = rn::find(self.nodes_, lhs, &node::id);
        const auto rhs_node_ptr = rn::find(self.nodes_, rhs, &node::id);

        if (lhs_node_ptr == self.nodes_.end() or rhs_node_ptr == self.nodes_.end()) {
            throw std::logic_error{
                "Trying to contract nodes whitch are not part of the network."
            };
        }

        const auto lhs_node = *lhs_node_ptr;
        const auto rhs_node = *rhs_node_ptr;

        auto is_lhs            = [&](const node& n) { return n.id == lhs; };
        auto is_rhs            = [&](const node& n) { return n.id == rhs; };
        auto is_partaker_node  = [&](const node& n) { return is_lhs(n) or is_rhs(n); };
        auto is_bystander_node = [&](const node& n) { return not is_partaker_node(n); };

        auto connected_to_lhs = [&](const edge& e) {
            return e.left.id == lhs or e.right.id == lhs;
        };
        auto connected_to_rhs = [&](const edge& e) {
            return e.left.id == rhs or e.right.id == rhs;
        };
        auto is_partaker_edge = [&](const edge& e) {
            return connected_to_lhs(e) or connected_to_rhs(e);
        };
        auto is_bystander_edge = [&](const edge& e) { return not is_partaker_edge(e); };

        const auto partaker_nodes =
            self.nodes_ | rv::filter(is_partaker_node) | rn::to<std::vector>();
        const auto bystander_nodes =
            self.nodes_ | rv::filter(is_bystander_node) | rn::to<std::vector>();
        const auto partaker_edges =
            self.edges_ | rv::filter(is_partaker_edge) | rn::to<std::vector>();
        const auto bystander_edges =
            self.edges_ | rv::filter(is_bystander_edge) | rn::to<std::vector>();

        self.nodes_ = bystander_nodes;
        self.edges_ = bystander_edges;

        // Now we have to just add nodes and edges which replace partakers.

        const auto partaker_contracted_edge = [&](const edge& e) {
            const auto left_partakes  = e.left.id == lhs or e.left.id == rhs;
            const auto right_partakes = e.right.id == lhs or e.right.id == rhs;
            return left_partakes and right_partakes;
        };

        const auto partaker_noncontracted_edge = [&](const edge& e) {
            return not partaker_contracted_edge(e);
        };

        const auto partaker_contracted_edges =
            partaker_edges | rv::filter(partaker_contracted_edge) | rn::to<std::vector>();
        const auto partaker_noncontracted_edges =
            partaker_edges | rv::filter(partaker_noncontracted_edge) | rn::to<std::vector>();

        const auto new_node_id = self.add_node(lhs_node.rank + rhs_node.rank
                                               - 2uz * rn::size(partaker_contracted_edges));

        // If i:th index of {l,r}hs is contracted, then i:th element here is empty optional.
        // If i:th index of {l,r}hs is free index, then i:th element is the index position
        // in the combined node.
        const auto [lhs_new_index_positions, rhs_new_index_positions] = std::invoke([&] {
            auto to_optionals =
                rv::transform([](const std::size_t i) { return std::optional{ i }; });

            using uz_opt_vec   = std::vector<std::optional<std::size_t>>;
            auto lhs_positions = rv::iota(0uz, lhs_node.rank) | to_optionals | rn::to<uz_opt_vec>();
            auto rhs_positions = rv::iota(lhs_node.rank) | rv::take(rhs_node.rank) | to_optionals
                                 | rn::to<uz_opt_vec>();

            for (const auto& e : partaker_contracted_edges) {
                if (e.left.id == lhs) {
                    lhs_positions[e.left.index] = std::nullopt;
                } else if (e.left.id == rhs) {
                    rhs_positions[e.left.index] = std::nullopt;
                } else {
                    throw std::logic_error{
                        "partaker_contracted_edges should connect to either lhs or rhs."
                    };
                }

                if (e.right.id == lhs) {
                    lhs_positions[e.right.index] = std::nullopt;
                } else if (e.right.id == rhs) {
                    rhs_positions[e.right.index] = std::nullopt;
                } else {
                    throw std::logic_error{
                        "partaker_contracted_edges should connect to either lhs or rhs."
                    };
                }
            }

            auto contractions_so_far = 0uz;
            for (auto& lhs_pos : lhs_positions) {
                if (not lhs_pos) {
                    ++contractions_so_far;
                } else {
                    lhs_pos.value() -= contractions_so_far;
                }
            }

            for (auto& rhs_pos : rhs_positions) {
                if (not rhs_pos) {
                    ++contractions_so_far;
                } else {
                    rhs_pos.value() -= contractions_so_far;
                }
            }

            return std::pair{ std::move(lhs_positions), std::move(rhs_positions) };
        });

        for (const edge& e : partaker_noncontracted_edges) {
            const auto [partaker_end, bystander_end] = std::invoke([&] {
                if (e.left.id == lhs or e.left.id == rhs) {
                    return e;
                } else if (e.right.id == lhs or e.right.id == rhs) {
                    return edge{ e.right, e.left };
                } else {
                    throw std::logic_error{
                        "Partaker non-contracted edge should be connected to one partaker node."
                    };
                }
            });

            const auto& pos_lookup =
                (partaker_end.id == lhs) ? lhs_new_index_positions : rhs_new_index_positions;

            self.add_edge(bystander_end, { new_node_id, pos_lookup[partaker_end.index].value() });
        }
        return new_node_id;
    }

    /// Optimized sequence based on the given dimension \p D.
    [[nodiscard]] constexpr rn::range auto pairwise_contraction_sequence(this auto&& self,
                                                                         const std::size_t D) {
        // These will be initialized by the first group because of the ~ infinite cost.
        auto best_sequence      = std::vector<pairwise_contraction_type>{};
        auto best_sequence_cost = static_cast<std::size_t>(-1);

        if (self.size() == 1uz) {
            // Threre can not be pairwise contractions for one node.
            return best_sequence;
        }

        const auto [node_pairs, edge_groups] = self.group_edges_pairwise();

        for (const auto i : rv::iota(0uz, rn::size(node_pairs))) {
            const auto lhs = node_pairs[i].first;
            const auto rhs = node_pairs[i].second;

            auto head            = pairwise_contraction_type(lhs, rhs, std::move(edge_groups[i]));
            const auto head_cost = head.cost(D);

            if (head_cost < best_sequence_cost) {
                auto contracted_cnet = self;
                const auto id        = contracted_cnet.pairwise_contraction(lhs.id, rhs.id);
                head.store_out(*rn::find(contracted_cnet.view_nodes(), id, &node::id));

                const auto tail      = contracted_cnet.pairwise_contraction_sequence(D);
                const auto tail_cost = contraction_cost(tail, D);

                const auto head_tail_cost = head_cost + tail_cost;

                if (head_tail_cost < best_sequence_cost) {
                    // no rv::concat in gcc 14 :(
                    best_sequence = std::vector{ head };
                    rn::copy(tail, std::back_inserter(best_sequence));

                    best_sequence_cost = head_tail_cost;
                }
            }
        }

        return best_sequence;
    }
};

[[nodiscard]] constexpr std::vector<connected_tensor_network>
    tensor_network::connected_components(this auto&& self) {
    // Each node is at least part of component consisting of itself.
    auto components = self.nodes_ | rv::transform([&](const auto n) {
                          auto net = connected_tensor_network{};
                          net.nodes_.push_back(n);
                          net.next_id_ = self.next_id_;
                          return net;
                      })
                      | rn::to<std::vector>();

    auto find_component = [&](const auto id) -> std::size_t {
        for (const auto& [i, c] : components | rv::enumerate) {
            if (c.contains(id)) { return i; }
        }
        throw std::logic_error{ "Each node should be part of some component." };
    };

    auto connect_components = [&](const edge& e) {
        const auto left_index  = find_component(e.left.id);
        const auto right_index = find_component(e.right.id);
        auto& left             = components[left_index];
        auto& right            = components[right_index];

        if (left_index == right_index) {
            // Edge connects connected component to itself,
            // so there is nothign to do,
            // but to pass this edge to the created connected_tensor_network.
            left.add_edge(e.left, e.right);
            return;
        }

        left.connect(right, e);
        components.erase(rn::next(components.begin(), right_index));
    };

    rn::for_each(self.edges_, connect_components);

    return components;
}

} // namespace idg
