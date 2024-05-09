
#include "../include/rt_incl.h"

#include <cstdlib>
#include <ctime>

int get_sm_count() {
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount,
                           device_id);
    return number_of_sm;
}

bool _is_tmp_name(std::string name) {
    return name.find("dollarbir") != std::string::npos;
}

void MEMORY_STAT() {
    auto mem_stat = rmm::available_device_memory();
    std::cout << " Available Memory: " << mem_stat.first
              << " Total Memory: " << mem_stat.second << std::endl;
}

void init_tmp_copy(LIE &lie, Relation *src, Relation *dst,
                   std::vector<int> indices, int grid_size, int block_size) {
    lie.add_ra(RelationalCopy(src, NEWT, dst, TupleProjector(indices),
                              grid_size, block_size, true));
}

bool metavar_wildcard_huh(column_meta_t metavar) {
    if (std::holds_alternative<std::string>(metavar)) {
        return std::get<std::string>(metavar).compare(WILDCARD) == 0;
    } else {
        return false;
    }
}

bool metavar_exists_huh(column_meta_t metavar,
                        std::vector<column_meta_t> &cols) {
    std::string metavar_name;
    if (std::holds_alternative<std::string>(metavar)) {
        metavar_name = std::get<std::string>(metavar);
    } else {
        return false;
    }
    for (auto &col : cols) {
        if (std::holds_alternative<std::string>(col)) {
            std::string target_mv = std::get<std::string>(col);
            if (target_mv.compare(metavar_name) == 0) {
                return true;
            }
        }
    }
    return false;
}

bool metavar_exists_huh(std::string metavar_name,
                        std::vector<std::string> &cols) {
    for (auto &col : cols) {
        if (col.compare(metavar_name) == 0) {
            return true;
        }
    }
    return false;
}

bool metavar_exists_huh(std::string metavar_name,
                        std::vector<column_meta_t> &cols) {
    for (auto &col : cols) {
        if (std::holds_alternative<std::string>(col)) {
            std::string target_mv = std::get<std::string>(col);
            if (target_mv.compare(metavar_name) == 0) {
                return true;
            }
        }
    }
    return false;
}

void lie_relations(LIE &lie, std::vector<Relation *> non_static_rels,
                   std::vector<Relation *> static_rels) {
    for (auto rel : non_static_rels) {
        lie.add_relations(rel, false);
    }
    for (auto rel : static_rels) {
        lie.add_relations(rel, true);
    }
}

void unify_join_col(std::vector<std::string> &inner,
                    std::vector<std::string> &outer,
                    std::vector<std::string> &output,
                    std::vector<int> &result) {
    for (auto &col : output) {
        // if col is a pure number, directly add it to result
        if (is_number(col)) {
            result.push_back(std::stoi(col));
            continue;
        }
        // if its a minus number
        if (col[0] == '-') {
            result.push_back(std::stoi(col));
            continue;
        }
        auto it = std::find(inner.begin(), inner.end(), col);
        if (it != inner.end()) {
            result.push_back(std::distance(inner.begin(), it));
        } else {
            it = std::find(outer.begin(), outer.end(), col);
            if (it != outer.end()) {
                result.push_back(std::distance(outer.begin(), it) +
                                 inner.size());
            } else {
                std::cout << "Outer: ";
                for (auto a : outer) {
                    std::cout << a << " ";
                }
                std::cout << std::endl;
                std::cout << "Inner: ";
                for (auto a : inner) {
                    std::cout << a << " ";
                }
                std::cout << std::endl;
                std::cout << "Output: ";
                for (auto a : output) {
                    std::cout << a << " ";
                }
                std::cout << std::endl;
                std::stringstream err_ss;
                err_ss << "Column not found in inner or outer: " << col;
                throw std::runtime_error(err_ss.str());
            }
        }
    }
}

// place all joined column at the front of inner/outer
// create order int vector for inner/outer
int reorder_index(std::string inner_name, std::vector<std::string> &inner,
                  std::string outer_name, std::vector<std::string> &outer,
                  std::string output_name, std::vector<std::string> &output,
                  std::vector<int> &inner_order,
                  std::vector<int> &outer_order) {
    std::vector<int> inner_joined;
    std::vector<int> inner_nonjoined;
    std::vector<int> outer_joined;
    std::vector<int> outer_nonjoined;
    for (int i = 0; i < inner.size(); i++) {
        auto col = inner[i];
        if (is_number(col)) {
            continue;
        }
        if (col.compare(WILDCARD) == 0) {
            continue;
        }
        auto it = std::find(outer.begin(), outer.end(), col);
        if (it != outer.end()) {
            inner_joined.push_back(i);
            outer_joined.push_back(std::distance(outer.begin(), it));
        } else {
            auto it_output = std::find(output.begin(), output.end(), col);
            if (it_output != output.end()) {
                inner_nonjoined.push_back(i);
            }
        }
    }

    assert(inner_joined.size() + inner_nonjoined.size() == inner.size());

    std::vector<std::string> inner_old = inner;
    int inner_used_cols_cnt = inner_joined.size() + inner_nonjoined.size();
    for (int i = 0; i < inner_joined.size(); i++) {
        inner[i] = inner_old[inner_joined[i]];
        inner_order.push_back(inner_joined[i]);
        // std::cout << "Inner " << i << " " << inner_joined[i] << inner[i] <<
        // std::endl;
    }

    for (int i = 0; i < inner_nonjoined.size(); i++) {
        inner[inner_joined.size() + i] = inner_old[inner_nonjoined[i]];
        inner_order.push_back(inner_nonjoined[i]);
    }

    std::cout << "Inner order " << inner_name << " : ";
    for (auto i : inner_order) {
        std::cout << "(" << i << "," << inner_old[i] << ") ";
    }
    std::cout << std::endl;

    // std::cout << "Inner reordered " << std::endl;

    for (int i = 0; i < outer.size(); i++) {
        auto col = outer[i];
        if (is_number(col)) {
            continue;
        }
        if (col.compare(WILDCARD) == 0) {
            continue;
        }
        auto it = std::find(inner.begin(), inner.end(), col);
        if (it == inner.end()) {
            auto it_output = std::find(output.begin(), output.end(), col);
            if (it_output != output.end()) {
                outer_nonjoined.push_back(i);
            }
        }
    }

    std::vector<std::string> outer_old = outer;
    if (!_is_tmp_name(outer_name)) {
        // algin outer with what in the inner
        int outer_used_cols_cnt = outer_joined.size() + outer_nonjoined.size();
        for (int i = 0; i < outer_joined.size(); i++) {
            // std::cout << "Outer " << i << " " << outer_joined[i] <<
            // std::endl;
            outer[i] = outer_old[outer_joined[i]];
            outer_order.push_back(outer_joined[i]);
        }
        for (int i = 0; i < outer_nonjoined.size(); i++) {
            outer[outer_joined.size() + i] = outer_old[outer_nonjoined[i]];
            outer_order.push_back(outer_nonjoined[i]);
        }
    } else {
        for (int i = 0; i < outer.size(); i++) {
            outer_order.push_back(i);
        }
    }

    std::cout << "Outer order " << outer_name << " : ";
    for (auto i : outer_order) {
        assert(outer_old[i] != "_");
        std::cout << "(" << i << "," << outer_old[i] << ") ";
    }
    std::cout << std::endl;

    std::cout << "Output " << output_name << " Canonical Order: ";
    for (auto i : output) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    // std::cout << "Outer reordered " << std::endl;
    return inner_joined.size();
}

TupleGenerator _unified_tpgen(std::vector<std::string> &inner,
                              std::vector<std::string> &outer,
                              std::vector<std::string> &output) {
    std::vector<int> unified_col;
    unify_join_col(inner, outer, output, unified_col);
    return TupleGenerator(inner.size(), unified_col);
}

TupleFilter _const_tpfilter(std::vector<column_meta_t> &target) {
    std::vector<BinaryFilterComparison> ops;
    std::vector<long> cols;
    std::vector<long> values;
    std::vector<std::string> meta_vars;
    std::vector<int> meta_vars_pos;
    for (int cnt = 0; cnt < target.size(); cnt++) {
        auto col = target[cnt];
        if (metavar_wildcard_huh(col)) {
            continue;
        }
        if (std::holds_alternative<std::string>(col)) {
            // check if there is the same meta variable in the target
            auto it = std::find(meta_vars.begin(), meta_vars.end(),
                                std::get<std::string>(col));
            if (it != meta_vars.end()) {
                std::cout << "Duplicate columns: " << std::get<std::string>(col)
                          << std::endl;
                ops.push_back(FOP(EQ));
                cols.push_back(
                    meta_vars_pos[std::distance(meta_vars.begin(), it)]);
                values.push_back(cnt);
            } else {
                meta_vars.push_back(std::get<std::string>(col));
                meta_vars_pos.push_back(cnt);
            }
        } else if (std::holds_alternative<long>(col)) {
            ops.push_back(FOP(EQ));
            cols.push_back(cnt);
            values.push_back(std::get<long>(col));
        }
    }
    return TupleFilter(ops, cols, values);
}

void TMP_CONST_FILTER(LIE &lie, Relation *rel,
                      std::vector<column_meta_t> target) {
    lie.add_ra(RelationalFilter(rel, NEWT, _const_tpfilter(target)));
}

void TMP_CONST_COPY_FILTER(LIE &lie, Relation *dest, Relation *from,
                           RelationVersion ver, std::vector<int> indices,
                           std::vector<column_meta_t> target) {
    auto sm = get_sm_count();
    auto block_size = 512;
    auto grid_size = 32 * sm;
    lie.add_ra(RelationalCopy(from, ver, dest, TupleProjector(indices),
                              grid_size, block_size, true));
    lie.add_ra(RelationalFilter(from, ver, _const_tpfilter(target)));
}

void TMP_COPY(LIE &lie, Relation *src, Relation *dst, int grid_size,
              int block_size) {
    int arity = src->arity;
    std::vector<int> indices(arity);
    for (int i = 0; i < arity; i++) {
        indices[i] = i;
    }
    lie.add_ra(RelationalCopy(src, NEWT, dst, TupleProjector(indices),
                              grid_size, block_size, true));
}

void BINARY_UNION(LIE &lie, Relation *src, RelationVersion ver1, Relation dst,
                  RelationVersion ver2) {
    GHashRelContainer *src_c = src->get_version(ver1);
    GHashRelContainer *dst_c = dst.get_version(ver2);
    lie.add_ra(RelationalUnion(src_c, dst_c));
}

void TMP_UNION(LIE &lie, Relation *src, Relation *dst) {
    GHashRelContainer *src_c = src->get_version(NEWT);
    GHashRelContainer *dst_c = dst->get_version(NEWT);
    lie.add_ra(RelationalUnion(src_c, dst_c));
}

bool _is_canonical_order(std::vector<int> &order) {
    for (int i = 0; i < order.size(); i++) {
        if (order[i] != i) {
            return false;
        }
    }
    return true;
}

// change const number into string for output clause in join
std::vector<std::string>
remove_const_output_clause(std::vector<column_meta_t> cols) {
    std::vector<std::string> result;
    for (auto &col : cols) {
        if (std::holds_alternative<std::string>(col)) {
            result.push_back(std::get<std::string>(col));
        } else {
            long num = std::get<long>(col);
            result.push_back(std::to_string(num));
        }
    }
    return result;
}

std::vector<column_meta_t>
str_vec_to_column_meta_t_vec(std::vector<std::string> cols) {
    std::vector<column_meta_t> result;
    for (auto &col : cols) {
        result.push_back(col);
    }
    return result;
}

void GENERAL_BINARY_JOIN(LIE &lie, LIE &lie_init, std::string inner_name,
                         RelationVersion inner_ver, std::string outer_name,
                         RelationVersion outer_ver, std::string output_name,
                         std::vector<std::string> inner_cols,
                         std::vector<std::string> outer_cols,
                         std::vector<std::string> output_cols, bool index,
                         bool negation_flag, bool debug) {
    if (inner_cols.size() == 0 || outer_cols.size() == 0) {
        throw std::runtime_error("Empty column for join");
    }
    auto sm = get_sm_count();
    auto block_size = 512;
    auto grid_size = 32 * sm;
    float *detail_time = new float[10];

    auto output_rel = lie.relation_name_map[output_name];

    std::vector<int> inner_order;
    std::vector<int> outer_order;
    // std::cout << "Inner cols: ";
    // for (auto &col : inner_cols) {
    //     std::cout << col << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "Outer cols: ";
    // for (auto &col : outer_cols) {
    //     std::cout << col << " ";
    // }
    // std::cout << std::endl;
    auto outer_col_original = outer_cols;
    auto inner_col_original = inner_cols;
    int joined_cnt =
        reorder_index(inner_name, inner_cols, outer_name, outer_cols,
                      output_name, output_cols, inner_order, outer_order);

    if (lie.relation_name_map.find(inner_name) == lie.relation_name_map.end()) {
        std::cerr << "Inner relation not found " << inner_name << std::endl;
    }
    Relation *inner_cannonical = lie.relation_name_map[inner_name];
    std::cout << "Inner cannonical name " << inner_name << std::endl;
    Relation *outer_cannonical = lie.relation_name_map[outer_name];
    std::cout << "Outer cannonical name " << outer_name << std::endl;
    std::cout << inner_cannonical->name << std::endl;

    // if in debug mode, we reload inner outer from disk
    // if (debug) {
    //     lie.remove_relation(inner_cannonical);
    //     lie.relation_name_map[inner_name] = decl_relation_input(
    //         lie, inner_name, inner_cannonical->arity,
    //         inner_cannonical->index, dataset_path, string_map, grid_size,
    //         block_size);
    // }

    // compute inner relation name
    std::stringstream inner_indexed_ss;

    if (joined_cnt == 1 && _is_canonical_order(inner_order)) {
        // std::cout << "No need name " << inner_cannonical->name  << std::endl;
        // no need name
        inner_indexed_ss << inner_cannonical->name;
    } else {
        if (_is_tmp_name(inner_cannonical->name)) {
            inner_indexed_ss << inner_cannonical->name;
        } else {
            inner_indexed_ss << inner_cannonical->name << "__";
            for (int i = 0; i < inner_order.size(); i++) {
                inner_indexed_ss << inner_order[i];
                if (i != inner_order.size() - 1) {
                    inner_indexed_ss << "_";
                }
            }
            inner_indexed_ss << "__" << joined_cnt;
        }
    }
    std::string inner_indexed_name = inner_indexed_ss.str();
    // std::cout << "Inner indexed name " << inner_indexed_name << std::endl;
    if (lie.relation_name_map.find(inner_indexed_name) ==
        lie.relation_name_map.end()) {
        std::cout << "Inner rel: " << inner_name << " ";
        for (auto &col : inner_col_original) {
            std::cout << col << " ";
        }
        std::cout << std::endl;
        throw std::runtime_error("Indexing missing for : " +
                                 inner_indexed_name);
    }
    Relation *inner_rel = lie.relation_name_map[inner_indexed_name];
    if (inner_rel == nullptr) {
        std::cout << "Inner indexed relation found but null"
                  << inner_indexed_name << std::endl;
        throw std::runtime_error("Inner indexed relation not found");
    }

    // compute outer relation name
    std::stringstream outer_indexed_ss;
    if (joined_cnt == 1 && _is_canonical_order(outer_order)) {
        // no need name
        // std::cout << "No need name " << outer_cannonical->name  << std::endl;
        outer_indexed_ss << outer_cannonical->name;
    } else {
        if (_is_tmp_name(outer_cannonical->name)) {
            outer_indexed_ss << outer_cannonical->name;
        } else {
            outer_indexed_ss << outer_cannonical->name << "__";
            for (int i = 0; i < outer_order.size(); i++) {
                outer_indexed_ss << outer_order[i];
                if (i != outer_order.size() - 1) {
                    outer_indexed_ss << "_";
                }
            }
            outer_indexed_ss << "__" << joined_cnt;
        }
    }

    std::string outer_indexed_name = outer_indexed_ss.str();
    // std::cout << "Outer indexed name " << outer_indexed_name << std::endl;
    if (lie.relation_name_map.find(outer_indexed_name) ==
        lie.relation_name_map.end()) {
        // tmp relation need be generate here, but in outer so can be a tmp
        // relation
        Relation *tmp_index_rel =
            new Relation(outer_indexed_name, outer_cannonical->arity, nullptr,
                         0, joined_cnt, 0, true);
        bool is_incremental = lie.is_output_relation(outer_cannonical->name);
        if (is_incremental && index) {
            lie.add_relations(tmp_index_rel, false);
            // lie_init.add_relations(tmp_index_rel, false);
            std::stringstream index_error_ss;
            index_error_ss << "Indexing missing for : " << outer_indexed_name;
            throw std::runtime_error(index_error_ss.str());
        } else {
            lie.add_tmp_relation(tmp_index_rel);
        }
        // project from canonical to indexed
        std::cout << "Generate Tmp Index for " << outer_name << " "
                  << outer_indexed_name << std::endl;
        std::cout << " Project " << outer_cannonical->name << " from (";
        for (auto &col : outer_col_original) {
            std::cout << col << " ";
        }
        std::cout << ") to (";
        for (auto &col : outer_cols) {
            std::cout << col << " ";
        }
        std::cout << ")" << std::endl;
        PROJECT(lie, outer_cannonical->name, outer_ver, outer_indexed_name,
                str_vec_to_column_meta_t_vec(outer_col_original),
                str_vec_to_column_meta_t_vec(outer_cols), debug);
        if (_is_tmp_name(outer_cannonical->name)) {
            std::cerr << "Outer indexed relation not found "
                      << outer_indexed_name << std::endl;
        }
        // mod outer ver to newt since we use tmp index
        outer_ver = NEWT;
    } else {
        if (lie.is_tmp_relation(outer_indexed_name)) {
            outer_ver = NEWT;
        }
    }
    Relation *outer_rel = lie.relation_name_map[outer_indexed_name];
    if (outer_rel == nullptr) {
        std::cout << "Outer indexed relation found but null"
                  << outer_indexed_name << std::endl;
        throw std::runtime_error("Outer indexed relation not found");
    }

    auto utp = _unified_tpgen(inner_cols, outer_cols, output_cols);
    std::cout << "Output " << output_name << " reorder mapping: ";
    for (int i = 0; i < utp.arity; i++) {
        std::cout << utp.reorder_map[i] << " ";
    }
    std::cout << std::endl;

    if (!negation_flag) {
        auto join_ra =
            RelationalJoin(inner_rel, inner_ver, outer_rel, outer_ver,
                           output_rel, utp, grid_size, block_size, detail_time);
        if (debug) {
            join_ra.debug_flag = 0;
        }
        lie.add_ra(join_ra);
    } else {
        if (outer_ver != NEWT) {
            throw std::runtime_error("Negation only support newt relation");
        }
        auto negate_ra = RelationalNegation(inner_rel, inner_ver, outer_rel,
                                            outer_ver, grid_size, block_size);
        negate_ra.left_flag = false;
        // auto copy_ra = RelationalCopy(outer, NEWT, output_rel, )
        if (debug) {
            negate_ra.debug_flag = 0;
        }
        lie.add_ra(negate_ra);
        PROJECT(lie, outer_indexed_name, NEWT, output_name,
                str_vec_to_column_meta_t_vec(outer_cols),
                str_vec_to_column_meta_t_vec(output_cols), debug);
    }
}

void BINARY_JOIN(LIE &lie, LIE &lie_init, std::string inner_name,
                 RelationVersion inner_ver, std::string outer_name,
                 RelationVersion outer_ver, std::string output_name,
                 std::vector<std::string> inner_cols,
                 std::vector<std::string> outer_cols,
                 std::vector<std::string> output_cols, bool index, bool debug) {
    GENERAL_BINARY_JOIN(lie, lie_init, inner_name, inner_ver, outer_name,
                        outer_ver, output_name, inner_cols, outer_cols,
                        output_cols, index, false, debug);
}

void NEGATE(LIE &lie, LIE &lie_init, std::string inner_name,
            RelationVersion inner_ver, std::string outer_name,
            RelationVersion outer_ver, std::string output_name,
            std::vector<std::string> inner_cols,
            std::vector<std::string> outer_cols,
            std::vector<std::string> output_cols, bool index, bool debug) {
    GENERAL_BINARY_JOIN(lie, lie_init, inner_name, inner_ver, outer_name,
                        outer_ver, output_name, inner_cols, outer_cols,
                        output_cols, index, true, debug);
}

void SEMI_NAIVE_BINARY_JOIN(LIE &lie, LIE &lie_init, Relation *inner_rel,
                            Relation *outer_rel, Relation *output_rel,
                            std::vector<std::string> inner_cols,
                            std::vector<std::string> outer_cols,
                            std::vector<std::string> output_cols, int grid_size,
                            int block_size, float *detail_time) {
    // lie.add_ra(
    //     RelationalJoin(inner_rel, DELTA, outer_rel, DELTA, output_rel,
    //                    _unified_tpgen(inner_cols, outer_cols, output_cols),
    //                    grid_size, block_size, detail_time));
    lie.add_ra(
        RelationalJoin(inner_rel, DELTA, outer_rel, FULL, output_rel,
                       _unified_tpgen(inner_cols, outer_cols, output_cols),
                       grid_size, block_size, detail_time));
    lie.add_ra(
        RelationalJoin(outer_rel, DELTA, inner_rel, FULL, output_rel,
                       _unified_tpgen(outer_cols, inner_cols, output_cols),
                       grid_size, block_size, detail_time));
}

void SEMI_NAIVE_BINARY_JOIN(LIE &lie, LIE &lie_init, std::string inner_rel_name,
                            std::string outer_rel_name,
                            std::string output_rel_name,
                            std::vector<std::string> inner_cols,
                            std::vector<std::string> outer_cols,
                            std::vector<std::string> output_cols, bool debug) {
    if (inner_cols.size() == 0 || outer_cols.size() == 0) {
        throw std::runtime_error("Empty column for join");
    }
    BINARY_JOIN(lie, lie_init, inner_rel_name, FULL, outer_rel_name, DELTA,
                output_rel_name, inner_cols, outer_cols, output_cols, true,
                debug);
    BINARY_JOIN(lie, lie_init, outer_rel_name, FULL, inner_rel_name, DELTA,
                output_rel_name, outer_cols, inner_cols, output_cols, true,
                debug);
    // TODO: do we need this?
    BINARY_JOIN(lie, lie_init, inner_rel_name, DELTA, outer_rel_name, DELTA,
                output_rel_name, inner_cols, outer_cols, output_cols, true,
                debug);
}

std::vector<int> _tpprojector_index(std::vector<column_meta_t> &input_column,
                                    std::vector<column_meta_t> &output_column) {
    std::vector<int> result;
    std::vector<std::string> used_meta_vars;
    for (int i = 0; i < output_column.size(); i++) {
        auto col = output_column[i];
        if (std::holds_alternative<std::string>(col)) {
            if (std::get<std::string>(col).compare("_") == 0) {
                std::cout << ">>>>>>>>>>>>>>>>>> Wildcard column: " << i
                          << std::endl;
                continue;
            } else {
                auto it =
                    std::find(input_column.begin(), input_column.end(), col);
                if (it != input_column.end()) {
                    if (std::find(used_meta_vars.begin(), used_meta_vars.end(),
                                  std::get<std::string>(col)) !=
                        used_meta_vars.end()) {
                        std::cout << "Duplicate columns: "
                                  << std::get<std::string>(col) << std::endl;
                    } else {
                        used_meta_vars.push_back(std::get<std::string>(col));
                        result.push_back(
                            std::distance(input_column.begin(), it));
                    }
                } else {
                    std::cerr
                        << "Missing columns: " << std::get<std::string>(col)
                        << std::endl;
                    throw std::runtime_error("Column not found in input");
                }
            }
        } else if (std::holds_alternative<long>(col)) {
            continue;
        }
    }
    return result;
}

inline TupleProjector _tpprojector(std::vector<column_meta_t> &input_column,
                                   std::vector<column_meta_t> &output_column) {
    return TupleProjector(_tpprojector_index(input_column, output_column));
}

// void PROJECT(LIE &lie, Relation *src, RelationVersion src_ver, Relation *dst,
//              std::vector<int> target_cols) {
//     auto sm = get_sm_count();
//     auto block_size = 512;
//     auto grid_size = 32 * sm;
//     lie.add_ra(RelationalCopy(src, src_ver, dst, TupleProjector(target_cols),
//                               grid_size, block_size, true));
// }

void PROJECT(LIE &lie, std::string src_rel_name, RelationVersion src_ver,
             std::string dst_rel_name, std::vector<column_meta_t> input_column,
             std::vector<column_meta_t> output_column, bool debug) {
    auto sm = get_sm_count();
    auto block_size = 512;
    auto grid_size = 32 * sm;
    auto src = lie.relation_name_map[src_rel_name];
    auto dst = lie.relation_name_map[dst_rel_name];
    auto ra = RelationalCopy(src, src_ver, dst,
                             _tpprojector(input_column, output_column),
                             grid_size, block_size, true, debug);
    if (debug) {
        ra.debug_flag = 0;
    }
    lie.add_ra(ra);
}

void FILTER_T(LIE &lie, Relation *rel, RelationVersion ver,
              std::vector<column_meta_t> target);

void FILTER_COPY(LIE &lie, std::string src_name, RelationVersion src_ver,
                 std::string dest_name, std::vector<column_meta_t> input_column,
                 std::vector<column_meta_t> output_column, bool debug,
                 bool tail) {
    auto src = lie.relation_name_map[src_name];
    auto dest = lie.relation_name_map[dest_name];
    auto tpf = _const_tpfilter(input_column);
    auto tppj = _tpprojector(input_column, output_column);
    auto ra = RelationalFilterProject(src, src_ver, tpf, dest, NEWT, tppj);
    if (debug) {
        ra.debug_flag = 0;
    }
    if (tail) {
        lie.add_tail_ra(ra);
    } else {
        lie.add_ra(ra);
    }
}

void ALIAS_RELATION(LIE &lie, LIE &lie_init, std::string rel_name,
                    std::string alias_name) {
    if (lie.relation_name_map.find(rel_name) == lie.relation_name_map.end()) {
        throw std::runtime_error("Relation not found for aliasing");
    }
    if (lie.relation_name_map.find(alias_name) != lie.relation_name_map.end()) {
        throw std::runtime_error("Alias name already exists");
    }
    auto rel = lie.relation_name_map[rel_name];
    lie.relation_name_map[alias_name] = rel;
    lie_init.relation_name_map[alias_name] = rel;
}

bool is_arithm_op(std::string op) {
    return op.compare("+") == 0 || op.compare("-") == 0 ||
           op.compare("*") == 0 || op.compare("/") == 0;
}

bool is_cmp_op(std::string op) {
    return op.compare("<") == 0 || op.compare(">") == 0 ||
           op.compare("<=") == 0 || op.compare(">=") == 0 ||
           op.compare("==") == 0 || op.compare("=/=") == 0;
}

BinaryFilterComparison get_cmp_op(std::string op) {
    if (op.compare("<") == 0) {
        return BinaryFilterComparison::LT;
    } else if (op.compare(">") == 0) {
        return BinaryFilterComparison::GT;
    } else if (op.compare("<=") == 0) {
        return BinaryFilterComparison::LE;
    } else if (op.compare(">=") == 0) {
        return BinaryFilterComparison::GE;
    } else if (op.compare("==") == 0) {
        return BinaryFilterComparison::EQ;
    } else if (op.compare("=/=") == 0) {
        return BinaryFilterComparison::NE;
    } else {
        throw std::runtime_error("Invalid comparison operator");
    }
}

// TODO: add max and min?
BinaryArithmeticOperator get_arithm_op(std::string op) {
    if (op.compare("+") == 0) {
        return BinaryArithmeticOperator::ADD;
    } else if (op.compare("-") == 0) {
        return BinaryArithmeticOperator::SUB;
    } else if (op.compare("*") == 0) {
        return BinaryArithmeticOperator::MUL;
    } else if (op.compare("/") == 0) {
        return BinaryArithmeticOperator::DIV;
    } else {
        throw std::runtime_error("Invalid arithmetic operator");
    }
}

bool is_negation(std::string name) { return name[0] == '~'; }

bool is_relname(std::string name) {
    return !is_negation(name) && !is_arithm_op(name) && !is_cmp_op(name);
}

template <typename BinaryOp>
std::string binary_op_to_string(std::vector<BinaryOp> ops);
template <>
std::string binary_op_to_string(std::vector<BinaryFilterComparison> ops) {
    std::string result;
    for (auto &op : ops) {
        switch (op) {
        case BinaryFilterComparison::LT:
            result += "<";
            break;
        case BinaryFilterComparison::GT:
            result += ">";
            break;
        case BinaryFilterComparison::LE:
            result += "<=";
            break;
        case BinaryFilterComparison::GE:
            result += ">=";
            break;
        case BinaryFilterComparison::EQ:
            result += "==";
            break;
        case BinaryFilterComparison::NE:
            result += "=/=";
            break;
        default:
            throw std::runtime_error("Invalid comparison operator");
        }
    }
    return result;
}
template <>
std::string binary_op_to_string(std::vector<BinaryArithmeticOperator> ops) {
    std::string result;
    for (auto &op : ops) {
        switch (op) {
        case BinaryArithmeticOperator::ADD:
            result += "+";
            break;
        case BinaryArithmeticOperator::SUB:
            result += "-";
            break;
        case BinaryArithmeticOperator::MUL:
            result += "*";
            break;
        case BinaryArithmeticOperator::DIV:
            result += "/";
            break;
        default:
            throw std::runtime_error("Invalid arithmetic operator");
        }
    }
    return result;
}

template <typename BinaryOp> BinaryOp get_binary_op(std::string in);
template <> BinaryFilterComparison get_binary_op(std::string in) {
    return get_cmp_op(in);
}
template <> BinaryArithmeticOperator get_binary_op(std::string in) {
    return get_arithm_op(in);
}

template <typename TupleHook, typename BinaryOp>
TupleHook _comp_tp(std::vector<column_meta_t> &target, clause_meta comp) {
    std::vector<BinaryOp> ops;
    std::vector<long> cols;
    std::vector<long> values;

    auto op = get_binary_op<BinaryOp>(comp.rel_name);
    ops.push_back(op);

    column_meta_t lhs = comp.columns[0];
    column_meta_t rhs = comp.columns[1];

    auto find_helper = [&](std::vector<column_meta_t> &target,
                           column_meta_t col) -> int {
        bool found = false;
        if (std::holds_alternative<long>(col)) {
            return std::get<long>(col);
        }
        for (int i = 0; i < target.size(); i++) {
            if (std::holds_alternative<std::string>(col)) {
                if (std::get<std::string>(col).compare(
                        std::get<std::string>(target[i])) == 0) {
                    found = true;
                    return i;
                }
            }
        }
        if (!found) {
            throw std::runtime_error("Column not found in target");
        }
        return -1;
    };
    // compute lhs pos
    long lhs_pos = find_helper(target, lhs);
    long rhs_pos = find_helper(target, rhs);
    cols.push_back(lhs_pos);
    values.push_back(rhs_pos);

    std::cout << "binary ops " << binary_op_to_string<BinaryOp>(ops)
              << " lhs pos: " << lhs_pos << " rhs pos: " << rhs_pos
              << std::endl;
    return TupleHook(ops, cols, values);
}

template <typename TupleHook, typename BinaryOp, typename BinaryRA>
void COMP_REL(LIE &lie, std::string src_rel_name, RelationVersion src_ver,
              std::string dest_rel_name,
              std::vector<column_meta_t> input_column,
              std::vector<column_meta_t> output_column, clause_meta filter_comp,
              bool debug) {
    auto src = lie.relation_name_map[src_rel_name];
    // check if src is in the relation map
    if (lie.relation_name_map.find(src_rel_name) ==
        lie.relation_name_map.end()) {
        std::string error_msg = "Relation not found: " + src_rel_name;
        throw std::runtime_error(error_msg);
    }
    auto dest = lie.relation_name_map[dest_rel_name];
    auto ft = _comp_tp<TupleHook, BinaryOp>(input_column, filter_comp);
    std::cout << "Comp from " << src_rel_name << " ";
    for (auto &col : input_column) {
        if (std::holds_alternative<std::string>(col)) {
            std::cout << std::get<std::string>(col) << " ";
        } else {
            std::cout << std::get<long>(col) << " ";
        }
    }
    std::cout << std::endl;
    std::string version_str = "FULL";
    if (src_ver == DELTA) {
        version_str = "DELTA";
    } else {
        version_str = "NEWT";
    }
    std::cout << version_str << " to " << dest_rel_name << " ";
    for (auto &col : output_column) {
        if (std::holds_alternative<std::string>(col)) {
            std::cout << std::get<std::string>(col) << " ";
        } else {
            std::cout << std::get<long>(col) << " ";
        }
    }
    std::cout << std::endl;
    auto prj = _tpprojector(input_column, output_column);
    auto ra = BinaryRA(src, src_ver, ft, dest, NEWT, prj);
    if (debug) {
        // std::vector<column_type> test_m = {};
        ra.debug_flag = 0;
    }
    lie.add_ra(ra);
}

// helper function to generate constant filter from a clause
std::string gen_tmp_filter_rel(LIE &lie, int rule_id, int clause_id,
                               clause_meta clause, RelationVersion ver,
                               int index_column_size = 0, int arity = -1) {
    auto sm = get_sm_count();
    auto block_size = 512;
    auto grid_size = 32 * sm;
    auto clause_rel_name = clause.rel_name;
    auto clause_rel = lie.relation_name_map[clause_rel_name];
    if (arity == -1) {
        arity = clause_rel->arity;
    }
    if (!clause.need_filter()) {
        return clause_rel_name;
    } else {
        std::stringstream tmp_rel_name_ss;
        // gen random number
        srand((unsigned)time(0));
        auto rd = (rand() % 2048) + 1;
        tmp_rel_name_ss << "dollarbir_rule" << rule_id << "_filter_delta_"
                        << clause_id << "_" << rd;
        std::string tmp_rel_name = tmp_rel_name_ss.str();
        if (index_column_size == 0) {
            Relation *tmp_rel = new Relation(tmp_rel_name, arity, nullptr, 0, 1,
                                             0, grid_size, block_size, true);
            lie.add_tmp_relation(tmp_rel);
        } else {
            // TODO: change arity
            Relation *tmp_rel =
                new Relation(tmp_rel_name, arity, nullptr, 0, index_column_size,
                             0, grid_size, block_size, false);
            lie.add_relations(tmp_rel, false);
        }

        return tmp_rel_name;
    }
}

// generate all combination of select `num` clause from `clauses`
void select_combination(std::vector<clause_meta> clauses, int num,
                        std::vector<std::vector<clause_meta>> &selected,
                        std::vector<std::vector<clause_meta>> &non_selected) {
    std::vector<bool> v(clauses.size());
    std::fill(v.end() - num, v.end(), true);
    do {
        std::vector<clause_meta> tmp;
        std::vector<clause_meta> non_tmp;
        for (int i = 0; i < v.size(); i++) {
            if (v[i]) {
                tmp.push_back(clauses[i]);
            } else {
                non_tmp.push_back(clauses[i]);
            }
        }
        selected.push_back(tmp);
        non_selected.push_back(non_tmp);
    } while (std::next_permutation(v.begin(), v.end()));
}

std::vector<std::string> force_interp_metavar(std::vector<column_meta_t> cols) {
    std::vector<std::string> result;
    for (auto &col : cols) {
        if (std::holds_alternative<std::string>(col)) {
            result.push_back(std::get<std::string>(col));
        } else {
            // exits and throw error
            throw std::runtime_error("const must be eliminated before join");
        }
    }
    return result;
}

void metavar_diff(std::vector<std::string> &cols1,
                  std::vector<std::string> &cols2,
                  std::vector<std::string> &result) {
    for (auto &col : cols1) {
        if (!metavar_exists_huh(col, cols2)) {
            result.push_back(col);
        }
    }
}

void process_copy_rule(LIE &lie, int rule_id,
                       std::vector<clause_meta> &incremental_clauses,
                       clause_meta copy_src_clause, clause_meta output_clause,
                       bool debug) {
    Relation *src_rel = lie.relation_name_map[copy_src_clause.rel_name];
    if (copy_src_clause.need_filter()) {
        // auto tmp_rel_name =
        //     gen_tmp_filter_rel(lie, rule_id, incremental_clauses.size(),
        //                        copy_src_clause, copy_src_clause.ver);
        // std::cout << "CopyFilter from " << copy_src_clause.rel_name << " "
        //           << copy_src_clause.ver << " to " << tmp_rel_name <<
        //           std::endl;
        FILTER_COPY(lie, copy_src_clause.rel_name, copy_src_clause.ver,
                    output_clause.rel_name, copy_src_clause.columns,
                    output_clause.columns, debug);
    } else {
        PROJECT(lie, copy_src_clause.rel_name, copy_src_clause.ver,
                output_clause.rel_name, copy_src_clause.columns,
                output_clause.columns, debug);
    }
}

void process_non_incremental_const_filter_clause(
    LIE &lie, LIE &lie_init, int rule_id, int nc_clause_cnt, clause_meta &ic,
    std::vector<std::vector<std::string>> &joined_columns_non_incr_clause,
    std::vector<std::vector<std::string>> &non_join_columns_non_incr_clause,
    bool debug) {
    auto sm = get_sm_count();
    auto block_size = 512;
    auto grid_size = 32 * sm;
    std::stringstream tmp_rel_name_ss;
    tmp_rel_name_ss << "dollarbir_rule" << rule_id << "_filer_full_"
                    << nc_clause_cnt;
    // << nc_clause_cnt + incremental_clauses.size();
    std::string tmp_rel_name = tmp_rel_name_ss.str();
    std::vector<column_meta_t> filtered_cols;
    int arity = 0;
    auto joinec_cs = joined_columns_non_incr_clause[nc_clause_cnt];
    auto non_joince_cs = non_join_columns_non_incr_clause[nc_clause_cnt];
    // place all joined column at the front
    for (auto &jc : joinec_cs) {
        filtered_cols.push_back(jc);
        arity++;
    }
    for (auto &njc : non_joince_cs) {
        // check if metavar exists in joined columns
        if (!metavar_exists_huh(njc, filtered_cols)) {
            filtered_cols.push_back(njc);
            arity++;
        }
    }
    auto jcc = joinec_cs.size();
    Relation *tmp_rel = new Relation(tmp_rel_name, arity, nullptr, 0, jcc, 0,
                                     grid_size, block_size, false);
    lie_init.add_relations(tmp_rel, false);
    lie.add_relations(tmp_rel, true);
    std::cout << "CopyFilter from " << ic.rel_name << " " << ic.ver << " to "
              << tmp_rel_name << std::endl;
    FILTER_COPY(lie_init, ic.rel_name, FULL, tmp_rel_name, ic.columns,
                filtered_cols, debug);
    // modify the clause to use the new relation
    ic.rel_name = tmp_rel_name;
    ic.columns = filtered_cols;
    ic.ver = FULL;
}

// join k incremental clauses together
void process_incremental_clauses_1(
    LIE &lie, LIE &lie_init, int rule_id, clause_meta &incr_outer,
    std::vector<clause_meta> &incremental_clauses,
    std::vector<std::vector<std::string>> &joined_columns_non_incr_clause,
    bool debug) {
    auto ic = incremental_clauses[0];
    if (ic.need_filter()) {
        // construct tmp relation metavar
        auto jcs = joined_columns_non_incr_clause[0];
        for (auto &jc : jcs) {
            incr_outer.columns.push_back(jc);
        }
        // non joined column
        for (auto &ic : ic.columns) {
            if (std::holds_alternative<std::string>(ic)) {
                if (metavar_wildcard_huh(ic)) {
                    continue;
                }
                auto metavar_name = std::get<std::string>(ic);
                if (!metavar_exists_huh(metavar_name, jcs)) {
                    incr_outer.columns.push_back(ic);
                }
            }
        }
        auto tmp_rel_name = gen_tmp_filter_rel(lie, rule_id, 0, ic, DELTA, 0,
                                               incr_outer.columns.size());
        incr_outer.rel_name = tmp_rel_name;
        incr_outer.ver = NEWT;
        // PROJECT(lie, ic.rel_name, DELTA, tmp_rel_name, ic.columns,
        //         incr_outer.columns, debug);
        std::cout << "CopyFilter from " << ic.rel_name << " " << ic.ver
                  << " to " << tmp_rel_name << std::endl;
        FILTER_COPY(lie, ic.rel_name, DELTA, tmp_rel_name, ic.columns,
                    incr_outer.columns, debug);
        // TODO: need index here?
        // lie.add_ra(RelationalIndex(lie.relation_name_map[tmp_rel_name],
        // NEWT));
    } else {
        std::cout << ">>>>> No need filter for " << ic.rel_name << std::endl;
        incr_outer = ic;
        incr_outer.ver = DELTA;
    }
}

void process_incremental_clauses_2(
    LIE &lie, LIE &lie_init, int rule_id, clause_meta &incr_outer,
    clause_meta output_clasue, std::vector<clause_meta> &incremental_clauses,
    std::vector<clause_meta> &non_incremental_clauses,
    std::vector<std::vector<std::string>> &joined_columns_non_incr_clause,
    std::vector<std::vector<std::string>> &non_join_columns_non_incr_clause,
    std::vector<std::vector<std::string>> &required_columns_after, bool debug) {
    // compute the join columns
    std::vector<column_meta_t> joined_mvs_incr;
    for (auto &col : incremental_clauses[0].columns) {
        if (metavar_exists_huh(col, incremental_clauses[1].columns) &&
            !metavar_wildcard_huh(col)) {
            joined_mvs_incr.push_back(col);
        }
    }
    // check if need constant filter
    for (int i = 0; i < 2; i++) {
        if (incremental_clauses[i].need_filter()) {
            std::vector<column_meta_t> filtered_cols;
            int arity = 0;
            for (auto &jc : joined_mvs_incr) {
                filtered_cols.push_back(jc);
                arity++;
            }
            for (auto &ic : incremental_clauses[i].columns) {
                if (std::holds_alternative<std::string>(ic)) {
                    if (!metavar_exists_huh(ic, filtered_cols) &&
                        !metavar_wildcard_huh(ic)) {
                        filtered_cols.push_back(ic);
                        arity++;
                    }
                }
            }
            auto tmp_rel_name = gen_tmp_filter_rel(
                lie, rule_id, i, incremental_clauses[i],
                incremental_clauses[i].ver, joined_mvs_incr.size(), arity);
            assert(filtered_cols.size() ==
                   lie.relation_name_map[tmp_rel_name]->arity);

            std::cout << "CopyFilter from " << incremental_clauses[i].rel_name
                      << " " << incremental_clauses[i].ver << " to "
                      << tmp_rel_name << " ";
            for (auto &col : filtered_cols) {
                if (std::holds_alternative<std::string>(col)) {
                    std::cout << std::get<std::string>(col) << " ";
                    if (metavar_wildcard_huh(col)) {
                        std::cerr << "Error: _ in filter" << std::endl;
                    }
                } else {
                    std::cout << std::get<long>(col) << " ";
                }
            }
            std::cout << std::endl;
            // filter FULL init
            lie_init.add_relations(lie.relation_name_map[tmp_rel_name], false);
            FILTER_COPY(lie_init, incremental_clauses[i].rel_name, FULL,
                        tmp_rel_name, incremental_clauses[i].columns,
                        filtered_cols, debug);

            FILTER_COPY(lie, incremental_clauses[i].rel_name, NEWT,
                        tmp_rel_name, incremental_clauses[i].columns,
                        filtered_cols, debug, true);
            incremental_clauses[i].rel_name = tmp_rel_name;
            incremental_clauses[i].columns = filtered_cols;
            incremental_clauses[i].ver = DELTA;
        }
    }
    if (incremental_clauses[0].columns.size() == 0) {
        throw std::runtime_error("No column in incremental clause 0");
    }
    if (incremental_clauses[1].columns.size() == 0) {
        throw std::runtime_error("No column in incremental clause 1");
    }

    if (non_incremental_clauses.size() == 0) {
        // no need for temporary relation
        SEMI_NAIVE_BINARY_JOIN(
            lie, lie_init, incremental_clauses[0].rel_name,
            incremental_clauses[1].rel_name, output_clasue.rel_name,
            force_interp_metavar(incremental_clauses[0].columns),
            force_interp_metavar(incremental_clauses[1].columns),
            remove_const_output_clause(output_clasue.columns), debug);
    } else {
        // need temporary relation
        auto first_non_recur = non_incremental_clauses[0];
        auto used_mvs_after = required_columns_after[0];
        // auto jcs = joined_columns_non_incr_clause[0];
        std::vector<column_meta_t> jcs;
        for (auto &non_clause : non_incremental_clauses) {
            for (auto &col : non_clause.columns) {
                if ((metavar_exists_huh(col, incremental_clauses[0].columns) ||
                     metavar_exists_huh(col, incremental_clauses[1].columns)) &&
                    !metavar_exists_huh(col, jcs)) {
                    jcs.push_back(col);
                }
            }
        }

        // print used_mvs_after
        std::cout << "Joined columns: ";
        for (auto &mv : jcs) {
            if (std::holds_alternative<std::string>(mv)) {
                std::cout << std::get<std::string>(mv) << " ";
            }
        }
        std::cout << std::endl;
        // tmp relation must place the column need to be joined in next step
        // in the front
        std::stringstream tmp_rel_name_ss;
        tmp_rel_name_ss << "dollarbir_rule" << rule_id << "_join_0";
        std::string tmp_rel_name = tmp_rel_name_ss.str();
        int arity = 0;
        std::vector<column_meta_t> tmp_cols;
        for (auto &jc : jcs) {
            tmp_cols.push_back(jc);
            arity++;
        }
        for (auto &ic : incremental_clauses[0].columns) {
            if (std::holds_alternative<std::string>(ic)) {
                auto metavar_name = std::get<std::string>(ic);
                if (!metavar_exists_huh(metavar_name, tmp_cols) &&
                    !metavar_wildcard_huh(ic) &&
                    metavar_exists_huh(metavar_name, used_mvs_after)) {
                    tmp_cols.push_back(ic);
                    arity++;
                }
            }
        }
        for (auto &ic : incremental_clauses[1].columns) {
            if (std::holds_alternative<std::string>(ic)) {
                auto metavar_name = std::get<std::string>(ic);
                if (!metavar_exists_huh(metavar_name, tmp_cols) &&
                    !metavar_wildcard_huh(ic) &&
                    metavar_exists_huh(metavar_name, used_mvs_after)) {
                    tmp_cols.push_back(ic);
                    arity++;
                }
            }
        }
        std::cout << "Generate tmp relation for join 0 " << tmp_rel_name << " "
                  << arity << std::endl;
        Relation *tmp_rel =
            new Relation(tmp_rel_name, arity, nullptr, 0, 1, 0, true);
        lie.add_tmp_relation(tmp_rel);
        SEMI_NAIVE_BINARY_JOIN(
            lie, lie_init, incremental_clauses[0].rel_name,
            incremental_clauses[1].rel_name, tmp_rel_name,
            force_interp_metavar(incremental_clauses[0].columns),
            force_interp_metavar(incremental_clauses[1].columns),
            remove_const_output_clause(tmp_cols), debug);
        incr_outer.rel_name = tmp_rel_name;
        incr_outer.columns = tmp_cols;
        incr_outer.ver = NEWT;
    }
}

void process_incremental_clause_3(
    LIE &lie, LIE &lie_init, int rule_id, clause_meta &incr_outer,
    clause_meta output_clasue, std::vector<clause_meta> &incremental_clauses,
    std::vector<clause_meta> &non_incremental_clauses,
    std::vector<std::vector<std::string>> &joined_columns_non_incr_clause,
    std::vector<std::vector<std::string>> &non_join_columns_non_incr_clause,
    std::vector<std::vector<std::string>> &required_columns_after, bool debug) {

    int block_size = 512;
    int grid_size = 32 * get_sm_count();
    // find all join columns
    std::vector<std::vector<column_meta_t>> joined_mvs_incr_list(3);

    // print incremental_clauses[0] columns

    for (auto &col : incremental_clauses[0].columns) {
        if (metavar_exists_huh(col, incremental_clauses[1].columns)) {
            joined_mvs_incr_list[0].push_back(col);
            joined_mvs_incr_list[1].push_back(col);
        }
    }

    for (auto &col : incremental_clauses[2].columns) {
        if (metavar_exists_huh(col, incremental_clauses[0].columns) ||
            metavar_exists_huh(col, incremental_clauses[1].columns)) {
            joined_mvs_incr_list[2].push_back(col);
        }
    }

    // check if need constant filter
    for (int i = 0; i < 3; i++) {
        auto &ic = incremental_clauses[i];
        if (ic.need_filter()) {
            auto tmp_rel_name = gen_tmp_filter_rel(
                lie, rule_id, i, ic, ic.ver, joined_mvs_incr_list[i].size());
            std::vector<column_meta_t> filtered_cols;
            int arity = 0;
            for (auto &jc : joined_mvs_incr_list[i]) {
                filtered_cols.push_back(jc);
                arity++;
            }
            for (auto &ic : ic.columns) {
                if (std::holds_alternative<std::string>(ic)) {
                    if (!metavar_exists_huh(ic, filtered_cols)) {
                        filtered_cols.push_back(ic);
                        arity++;
                    }
                }
            }
            std::cout << "CopyFilter from " << ic.rel_name << " " << ic.ver
                      << " to " << tmp_rel_name << std::endl;
            FILTER_COPY(lie, ic.rel_name, ic.ver, tmp_rel_name, ic.columns,
                        filtered_cols, debug);
            ic.rel_name = tmp_rel_name;
            ic.columns = filtered_cols;
            ic.ver = NEWT;
            // index it, if its not outermost
            if (i != 0) {
                auto tmp_rel = lie.relation_name_map[tmp_rel_name];
                lie.add_ra(RelationalIndex(tmp_rel, NEWT));
            }
        }
    }
    // join 0 1
    // compute temp relation for join of 0 1
    std::vector<column_meta_t> tmp1_cols;
    for (auto &jc : joined_mvs_incr_list[1]) {
        tmp1_cols.push_back(jc);
    }
    if (non_incremental_clauses.size() == 0) {
        for (int i = 0; i < 2; i++) {
            for (auto &ic : incremental_clauses[i].columns) {
                if (metavar_exists_huh(ic, output_clasue.columns) &&
                    !metavar_wildcard_huh(ic) &&
                    !metavar_exists_huh(ic, tmp1_cols)) {
                    tmp1_cols.push_back(ic);
                }
            }
        }
    } else {
        // print required_columns_after[0]
        for (auto &ic : incr_outer.columns) {
            if (std::holds_alternative<std::string>(ic)) {
                auto mv_name = std::get<std::string>(ic);
                if (!metavar_exists_huh(ic, incremental_clauses[2].columns) &&
                    !metavar_wildcard_huh(ic) &&
                    !metavar_exists_huh(mv_name, tmp1_cols)) {
                    tmp1_cols.push_back(ic);
                }
            }
        }
    }

    std::cout << "joined_mvs_incr_list 0 columns: ";
    for (auto &col : joined_mvs_incr_list[0]) {
        if (std::holds_alternative<std::string>(col)) {
            std::cout << std::get<std::string>(col) << " ";
        } else {
            std::cout << std::get<long>(col) << " ";
        }
    }
    std::cout << std::endl;

    std::stringstream tmp_rel_name_ss;
    // gen a random number

    srand((unsigned)time(0));
    auto rd1 = (rand() % 2048) + 1;
    tmp_rel_name_ss << "dollarbir_rule" << rule_id << "_join_0_1_" << rd1;
    std::string tmp_rel_name = tmp_rel_name_ss.str();
    Relation *tmp_rel = new Relation(tmp_rel_name, tmp1_cols.size(), nullptr, 0,
                                     joined_mvs_incr_list[1].size(), 0, true);
    lie.add_tmp_relation(tmp_rel);
    BINARY_JOIN(lie, lie_init, incremental_clauses[1].rel_name,
                incremental_clauses[1].ver, incremental_clauses[0].rel_name,
                incremental_clauses[0].ver, tmp_rel_name,
                force_interp_metavar(incremental_clauses[1].columns),
                force_interp_metavar(incremental_clauses[0].columns),
                remove_const_output_clause(tmp1_cols), true, debug);

    // join tmp with 2
    if (non_incremental_clauses.size() == 0) {
        // no need for temporary relation
        BINARY_JOIN(lie, lie_init, incremental_clauses[2].rel_name,
                    incremental_clauses[2].ver, tmp_rel_name, NEWT,
                    output_clasue.rel_name,
                    force_interp_metavar(incremental_clauses[2].columns),
                    force_interp_metavar(tmp1_cols),
                    remove_const_output_clause(output_clasue.columns), true,
                    debug);
    } else {
        // use tmp relation for join 0 1 2
        BINARY_JOIN(
            lie, lie_init, incremental_clauses[2].rel_name,
            incremental_clauses[2].ver, tmp_rel_name, NEWT, incr_outer.rel_name,
            force_interp_metavar(incremental_clauses[2].columns),
            force_interp_metavar(tmp1_cols),
            remove_const_output_clause(incr_outer.columns), true, debug);
    }
}

void process_non_incremental_clauses(
    LIE &lie, LIE &lie_init, int rule_id, clause_meta &incr_outer,
    clause_meta output_clasue, std::vector<clause_meta> &incremental_clauses,
    std::vector<clause_meta> &non_incremental_clauses,
    std::vector<std::vector<std::string>> &joined_columns_non_incr_clause,
    std::vector<std::vector<std::string>> &required_columns_after, bool debug) {

    // join all non incremental clauses
    // clause_id = incremental_clauses.size();
    for (int i = 0; i < non_incremental_clauses.size(); i++) {
        auto nic = non_incremental_clauses[i];
        if (i != non_incremental_clauses.size() - 1) {
            // need temporary relation
            auto join_cols = joined_columns_non_incr_clause[i + 1];
            // auto used_mvs_after = required_columns_after[i + 1];
            std::vector<std::string> used_mvs_after;
            for (int j = i + 1; j < non_incremental_clauses.size(); j++) {
                for (auto &col : non_incremental_clauses[j].columns) {
                    if (std::holds_alternative<std::string>(col)) {
                        auto mv = std::get<std::string>(col);
                        if ((metavar_exists_huh(mv, incr_outer.columns) ||
                             metavar_exists_huh(
                                 mv, non_incremental_clauses[i].columns)) &&
                            !metavar_exists_huh(mv, used_mvs_after)) {
                            used_mvs_after.push_back(mv);
                        }
                    }
                }
            }
            for (auto &col : output_clasue.columns) {
                if (std::holds_alternative<std::string>(col)) {
                    auto mv = std::get<std::string>(col);
                    if ((metavar_exists_huh(mv, incr_outer.columns) ||
                         metavar_exists_huh(
                             mv, non_incremental_clauses[i].columns)) &&
                        !metavar_exists_huh(mv, used_mvs_after)) {
                        used_mvs_after.push_back(mv);
                    }
                }
            }
            // print used_mvs_after
            std::cout << "Used mvs after: ";
            for (auto &mv : used_mvs_after) {
                std::cout << mv << " ";
            }
            std::cout << std::endl;

            auto tmp_columns = std::vector<column_meta_t>();
            int arity = 0;
            // the metavar of tmp must contains all columns which is appears
            // in current outer and inner meanwhile also appears in clauses
            // after this
            for (auto &col : join_cols) {
                tmp_columns.push_back(col);
                arity++;
            }
            if (is_relname(nic.rel_name)) {
                for (auto &col : nic.columns) {
                    if (std::holds_alternative<std::string>(col)) {
                        auto mv = std::get<std::string>(col);
                        if (!metavar_exists_huh(col, tmp_columns) &&
                            !metavar_wildcard_huh(col) &&
                            metavar_exists_huh(mv, used_mvs_after)) {
                            tmp_columns.push_back(col);
                            arity++;
                        }
                    }
                }
            }
            for (auto &oc : incr_outer.columns) {
                if (std::holds_alternative<std::string>(oc)) {
                    auto mv = std::get<std::string>(oc);
                    if (!metavar_exists_huh(oc, tmp_columns) &&
                        !metavar_wildcard_huh(oc) &&
                        metavar_exists_huh(mv, used_mvs_after)) {
                        tmp_columns.push_back(oc);
                        arity++;
                    }
                }
            }
            std::stringstream tmp_rel_name_ss;
            tmp_rel_name_ss << "dollarbir_rule" << rule_id << "_join_"
                            << i + incremental_clauses.size();
            std::string tmp_rel_name = tmp_rel_name_ss.str();
            Relation *tmp_rel = new Relation(tmp_rel_name, arity, nullptr, 0,
                                             join_cols.size(), 0, true);
            std::cout << "Generate tmp relation for join " << i << " "
                      << tmp_rel_name << " " << arity << " ";
            for (auto &col : tmp_columns) {
                if (std::holds_alternative<std::string>(col)) {
                    std::cout << std::get<std::string>(col) << " ";
                } else {
                    std::cout << std::get<long>(col) << " ";
                }
            }
            std::cout << std::endl;
            lie.add_tmp_relation(tmp_rel);
            if (is_relname(nic.rel_name)) {
                if (!nic.negation_flag) {
                    BINARY_JOIN(lie, lie_init, nic.rel_name, nic.ver,
                                incr_outer.rel_name, incr_outer.ver,
                                tmp_rel_name, force_interp_metavar(nic.columns),
                                force_interp_metavar(incr_outer.columns),
                                remove_const_output_clause(tmp_columns), false,
                                debug);
                } else {
                    //  negation
                    std::cout << "Negation join " << nic.rel_name << " "
                              << nic.ver << " " << incr_outer.rel_name << " "
                              << incr_outer.ver << " " << tmp_rel_name << " "
                              << std::endl;
                    NEGATE(lie, lie_init, nic.rel_name, nic.ver,
                           incr_outer.rel_name, incr_outer.ver, tmp_rel_name,
                           force_interp_metavar(nic.columns),
                           force_interp_metavar(incr_outer.columns),
                           remove_const_output_clause(tmp_columns), false,
                           debug);
                }
            } else if (is_cmp_op(nic.rel_name)) {
                // TODO: check if next clause is join, if it is, no need for tmp
                // relation
                COMP_REL<TupleFilter, BinaryFilterComparison,
                         RelationalFilterProject>(
                    lie, incr_outer.rel_name, incr_outer.ver, tmp_rel_name,
                    incr_outer.columns, tmp_columns, nic, debug);
            } else if (is_arithm_op(nic.rel_name)) {
                // TODO: add arithm op
                COMP_REL<TupleArithmeticSingle, BinaryArithmeticOperator,
                         RelationalArithmProject>(
                    lie, incr_outer.rel_name, incr_outer.ver, tmp_rel_name,
                    incr_outer.columns, tmp_columns, nic, debug);
            } else {
                throw std::runtime_error("Invalid non incremental clause");
            }
            incr_outer.rel_name = tmp_rel_name;
            incr_outer.columns = tmp_columns;
            incr_outer.ver = NEWT;
        } else {
            // no need for temporary relation
            if (is_relname(nic.rel_name)) {
                if (!nic.negation_flag) {
                    BINARY_JOIN(
                        lie, lie_init, nic.rel_name, nic.ver,
                        incr_outer.rel_name, incr_outer.ver,
                        output_clasue.rel_name,
                        force_interp_metavar(nic.columns),
                        force_interp_metavar(incr_outer.columns),
                        remove_const_output_clause(output_clasue.columns),
                        false, debug);
                } else {
                    // negation
                    std::cout << "Negation join " <<
                        nic.rel_name << " " << nic.ver << " " <<
                        incr_outer.rel_name << " " << incr_outer.ver << " " <<
                        output_clasue.rel_name << std::endl; 
                    NEGATE(lie, lie_init, nic.rel_name, nic.ver,
                           incr_outer.rel_name, incr_outer.ver,
                           output_clasue.rel_name,
                           force_interp_metavar(nic.columns),
                           force_interp_metavar(incr_outer.columns),
                           remove_const_output_clause(output_clasue.columns),
                           false, debug);
                }
            } else if (is_cmp_op(nic.rel_name)) {
                COMP_REL<TupleFilter, BinaryFilterComparison,
                         RelationalFilterProject>(
                    lie, incr_outer.rel_name, incr_outer.ver,
                    output_clasue.rel_name, incr_outer.columns,
                    output_clasue.columns, nic, debug);
            } else if (is_arithm_op(nic.rel_name)) {
                COMP_REL<TupleArithmeticSingle, BinaryArithmeticOperator,
                         RelationalArithmProject>(
                    lie, incr_outer.rel_name, incr_outer.ver,
                    output_clasue.rel_name, incr_outer.columns,
                    output_clasue.columns, nic, debug);
            } else {
                throw std::runtime_error("Invalid non incremental clause");
            }
        }
    }
}

void DATALOG_RECURISVE_RULE(LIE &lie, LIE &lie_init, int rule_id,
                            std::vector<clause_meta> input_clauses,
                            clause_meta output_clause, bool debug) {
    std::cout << ">>>>>>>>>>> Processing rule " << rule_id << std::endl;
    std::string out_rel_name = output_clause.rel_name;
    auto out_rel = lie.relation_name_map[out_rel_name];
    auto output_columns = output_clause.columns;

    auto sm = get_sm_count();
    auto block_size = 512;
    auto grid_size = 32 * sm;

    // lie.is_output_relation(out_rel_name);
    // reorder all rule, place all incremental clause at first
    std::vector<clause_meta> incremental_clauses;
    std::vector<clause_meta> non_incremental_clauses;
    for (auto &clause : input_clauses) {
        std::string clause_rel_name = clause.rel_name;
        // if (!is_relname(clause_rel_name)) {
        //     continue;
        // }
        if (lie.is_output_relation(clause_rel_name)) {
            clause.ver = DELTA;
            incremental_clauses.push_back(clause);
        } else {
            clause.ver = FULL;
            non_incremental_clauses.push_back(clause);
        }
    }

    std::cout << "Incremental clauses: " << incremental_clauses.size()
              << " Non incremental clauses: " << non_incremental_clauses.size()
              << std::endl;

    std::vector<std::vector<std::string>> required_columns_after;
    // std::vector<std::string> joined_columns_each_non_incr_clause;
    for (int i = non_incremental_clauses.size() - 1; i >= 0; i--) {
        std::vector<std::string> req_cs;
        // std::cout << "Clause " << i << std::endl;
        if (i == non_incremental_clauses.size() - 1) {
            for (auto &ic : output_clause.columns) {
                if (std::holds_alternative<std::string>(ic)) {
                    req_cs.push_back(std::get<std::string>(ic));
                }
            }
        } else {
            req_cs = required_columns_after.back();
            for (auto &ic : non_incremental_clauses[i + 1].columns) {
                if (std::holds_alternative<std::string>(ic)) {
                    // if haven't seen this metavar before
                    if (!metavar_exists_huh(std::get<std::string>(ic),
                                            req_cs)) {
                        req_cs.push_back(std::get<std::string>(ic));
                    }
                }
            }
        }
        required_columns_after.push_back(req_cs);
    }

    // populate all metavar name used in incremental clauses
    std::vector<std::string> metavars_incr;
    for (auto &ic : incremental_clauses) {
        for (auto &col : ic.columns) {
            if (std::holds_alternative<std::string>(col)) {
                if (!metavar_exists_huh(std::get<std::string>(col),
                                        metavars_incr)) {
                    metavars_incr.push_back(std::get<std::string>(col));
                }
            }
        }
    }
    // populate all metavar name used in non incremental clauses
    std::vector<std::string> metavars_non_incr;
    for (auto &ic : non_incremental_clauses) {
        for (auto &col : ic.columns) {
            if (std::holds_alternative<std::string>(col)) {
                if (!metavar_exists_huh(std::get<std::string>(col),
                                        metavars_non_incr)) {
                    metavars_non_incr.push_back(std::get<std::string>(col));
                }
            }
        }
    }

    std::vector<std::vector<std::string>> required_columns_before;
    for (int i = 0; i < non_incremental_clauses.size(); i++) {
        std::vector<std::string> req_cs;
        if (i == 0) {
            req_cs = metavars_incr;
        } else {
            req_cs = required_columns_before[i - 1];
            for (auto &ic : non_incremental_clauses[i - 1].columns) {
                if (std::holds_alternative<std::string>(ic)) {
                    // if haven't seen this metavar before
                    if (!metavar_exists_huh(std::get<std::string>(ic),
                                            req_cs)) {
                        req_cs.push_back(std::get<std::string>(ic));
                    }
                }
            }
        }
        required_columns_before.push_back(req_cs);
    }

    // populated joined column in every non incremental clause
    std::vector<std::vector<std::string>> joined_columns_non_incr_clause;
    std::vector<std::vector<std::string>> non_join_columns_non_incr_clause;
    for (int i = 0; i < non_incremental_clauses.size(); i++) {
        std::vector<std::string> joined_cols;
        std::vector<std::string> non_join_cols;
        for (auto &ic : non_incremental_clauses[i].columns) {
            if (metavar_wildcard_huh(ic)) {
                continue;
            }
            if (std::holds_alternative<std::string>(ic)) {
                // joined_cols.push_back(std::get<std::string>(ic));
                std::string metavar_name = std::get<std::string>(ic);
                if (metavar_exists_huh(metavar_name,
                                       required_columns_before[i])) {
                    joined_cols.push_back(metavar_name);
                } else {
                    non_join_cols.push_back(metavar_name);
                }
            }
        }
        joined_columns_non_incr_clause.push_back(joined_cols);
        non_join_columns_non_incr_clause.push_back(non_join_cols);
    }

    // generate static constant filter for all non incremental clauses
    int nc_clause_cnt = 0;
    for (auto &ic : non_incremental_clauses) {
        if (!is_relname(ic.rel_name)) {
            // computational relation don't need constant filter
            continue;
        }
        if (ic.need_filter()) {
            process_non_incremental_const_filter_clause(
                lie, lie_init, rule_id, nc_clause_cnt, ic,
                joined_columns_non_incr_clause,
                non_join_columns_non_incr_clause, debug);
        }
        nc_clause_cnt++;
    }

    // copy
    if (non_incremental_clauses.size() == 0 &&
        incremental_clauses.size() == 1) {
        // check if need constant filter
        auto ic = incremental_clauses[0];
        ic.ver = DELTA;
        process_copy_rule(lie, rule_id, incremental_clauses, ic, output_clause,
                          debug);
        // its the only clause, no need to join
        return;
    }

    // join all incremental clauses
    // std::string incremental_part_name;
    // clause_meta tmp_ic;
    clause_meta incr_outer;

    if (incremental_clauses.size() == 1) {
        process_incremental_clauses_1(lie, lie_init, rule_id, incr_outer,
                                      incremental_clauses,
                                      joined_columns_non_incr_clause, debug);
    }

    // ignore const in output clause
    std::vector<std::string> output_metavar;
    for (int i = 0; i < output_columns.size(); i++) {
        if (std::holds_alternative<std::string>(output_columns[i])) {
            output_metavar.push_back(std::get<std::string>(output_columns[i]));
        }
    }

    // semi-navie evaluation
    // TODO: do not hand unroll here, its unnecessary
    if (incremental_clauses.size() == 2) {
        process_incremental_clauses_2(
            lie, lie_init, rule_id, incr_outer, output_clause,
            incremental_clauses, non_incremental_clauses,
            joined_columns_non_incr_clause, non_join_columns_non_incr_clause,
            required_columns_after, debug);
    }

    if (incremental_clauses.size() == 3) {
        for (auto nic : joined_columns_non_incr_clause[0]) {
            incr_outer.columns.push_back(nic);
        }
        for (auto ic : metavars_incr) {
            if (!metavar_exists_huh(ic, incr_outer.columns) &&
                !metavar_wildcard_huh(ic) &&
                (metavar_exists_huh(ic, metavars_non_incr) ||
                 metavar_exists_huh(ic, output_clause.columns))) {
                incr_outer.columns.push_back(ic);
            }
        }
        std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> incr_outer columns: "
                  << std::endl;
        for (auto ic : incr_outer.columns) {
            if (std::holds_alternative<std::string>(ic)) {
                std::cout << std::get<std::string>(ic) << " ";
            } else {
                std::cout << std::get<long>(ic) << " ";
            }
        }
        std::cout << std::endl;
        // for (auto nic : non_incremental_clauses[0].columns) {
        //     if (std::holds_alternative<std::string>(nic)) {
        //         auto metavar_name = std::get<std::string>(nic);
        //         if (!metavar_exists_huh(metavar_name, incr_outer.columns) &&
        //             metavar_exists_huh(metavar_name, metavars_incr)) {
        //             incr_outer.columns.push_back(nic);
        //         }
        //     }
        // }
        incr_outer.ver = NEWT;
        // gen a rel name for incr outer
        std::stringstream incr_outer_ss;
        incr_outer_ss << "dollarbir_rule" << rule_id << "_incr_outer";
        incr_outer.rel_name = incr_outer_ss.str();
        Relation *incr_outer_rel =
            new Relation(incr_outer.rel_name, incr_outer.columns.size(),
                         nullptr, 0, joined_columns_non_incr_clause[0].size(),
                         0, grid_size, block_size, false);
        lie.add_relations(incr_outer_rel, true);

        std::vector<std::vector<clause_meta>> incremental_clauses_orders;
        auto original_incremental_clauses = incremental_clauses;
        incremental_clauses[0] = original_incremental_clauses[0];
        incremental_clauses[1] = original_incremental_clauses[1];
        incremental_clauses[2] = original_incremental_clauses[2];
        incremental_clauses[0].ver = DELTA;
        incremental_clauses[1].ver = FULL;
        incremental_clauses[2].ver = FULL;
        incremental_clauses_orders.push_back(incremental_clauses);
        incremental_clauses[0] = original_incremental_clauses[1];
        incremental_clauses[1] = original_incremental_clauses[0];
        incremental_clauses[2] = original_incremental_clauses[2];
        incremental_clauses[0].ver = FULL;
        incremental_clauses[1].ver = DELTA;
        incremental_clauses[2].ver = FULL;
        incremental_clauses_orders.push_back(incremental_clauses);
        incremental_clauses[0] = original_incremental_clauses[1];
        incremental_clauses[1] = original_incremental_clauses[2];
        incremental_clauses[2] = original_incremental_clauses[0];
        incremental_clauses[0].ver = FULL;
        incremental_clauses[1].ver = FULL;
        incremental_clauses[2].ver = DELTA;
        incremental_clauses_orders.push_back(incremental_clauses);
        incremental_clauses[0] = original_incremental_clauses[0];
        incremental_clauses[1] = original_incremental_clauses[1];
        incremental_clauses[2] = original_incremental_clauses[2];
        incremental_clauses[0].ver = DELTA;
        incremental_clauses[1].ver = DELTA;
        incremental_clauses[2].ver = FULL;
        incremental_clauses_orders.push_back(incremental_clauses);
        incremental_clauses[0] = original_incremental_clauses[0];
        incremental_clauses[1] = original_incremental_clauses[1];
        incremental_clauses[2] = original_incremental_clauses[2];
        incremental_clauses[0].ver = DELTA;
        incremental_clauses[1].ver = FULL;
        incremental_clauses[2].ver = DELTA;
        incremental_clauses_orders.push_back(incremental_clauses);
        incremental_clauses[0] = original_incremental_clauses[1];
        incremental_clauses[1] = original_incremental_clauses[0];
        incremental_clauses[2] = original_incremental_clauses[2];
        incremental_clauses[0].ver = FULL;
        incremental_clauses[1].ver = DELTA;
        incremental_clauses[2].ver = DELTA;
        incremental_clauses_orders.push_back(incremental_clauses);
        incremental_clauses[0] = original_incremental_clauses[0];
        incremental_clauses[1] = original_incremental_clauses[1];
        incremental_clauses[2] = original_incremental_clauses[2];
        incremental_clauses[0].ver = DELTA;
        incremental_clauses[1].ver = DELTA;
        incremental_clauses[2].ver = DELTA;
        incremental_clauses_orders.push_back(incremental_clauses);
        for (auto ordered : incremental_clauses_orders) {
            process_incremental_clause_3(
                lie, lie_init, rule_id, incr_outer, output_clause, ordered,
                non_incremental_clauses, joined_columns_non_incr_clause,
                non_join_columns_non_incr_clause, required_columns_after,
                debug);
        }
    }

    if (incremental_clauses.size() > 3) {
        std::cout << "Rule " << rule_id << " not support clauses has "
                  << incremental_clauses.size() << " incremental clauses"
                  << std::endl;
        throw std::runtime_error("Not support more than 2 incremental clauses");
    }

    // join all non incremental clauses
    process_non_incremental_clauses(
        lie, lie_init, rule_id, incr_outer, output_clause, incremental_clauses,
        non_incremental_clauses, joined_columns_non_incr_clause,
        required_columns_after, debug);
}
