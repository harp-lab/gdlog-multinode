
#pragma once

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <rmm/cuda_device.hpp>
#include <sstream>
#include <stdlib.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <vector>

#include "../include/builtin.h"
#include "../include/exception.cuh"
#include "../include/lie.cuh"
#include "../include/print.cuh"
#include "../include/timer.cuh"

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#define STRING(S) #S

#define FOP(x) BinaryFilterComparison::x
#define CONCAT_(x, y) x_##y
#define CONCAT(x, y) CONCAT_(x, y)
#define WILDCARD "_"

#define ENVIRONMENT_INIT                                                       \
    auto dataset_path = argv[1];                                               \
    std::map<column_type, std::string> string_map;                             \
    std::map<std::string, Relation *> relation_map;                            \
    float detail_time[10];

#define INIT_VER(NAME) NAME##_init

#define MAIN_ENTRANCE(RUN)                                                     \
    int main(int argc, char *argv[]) {                                         \
        int device_id;                                                         \
        int number_of_sm;                                                      \
        cudaGetDevice(&device_id);                                             \
        cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount,  \
                               device_id);                                     \
        int max_threads_per_block;                                             \
        cudaDeviceGetAttribute(&max_threads_per_block,                         \
                               cudaDevAttrMaxThreadsPerBlock, 0);              \
        int block_size, grid_size;                                             \
        block_size = 512;                                                      \
        grid_size = 32 * number_of_sm;                                         \
        std::locale loc("");                                                   \
        rmm::mr::cuda_memory_resource cuda_mr{};                               \
        rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{       \
            &cuda_mr, 4 * 256 * 1024};                                         \
        rmm::mr::set_current_device_resource(&mr);                             \
        RUN(argc, argv, block_size, grid_size);                                \
        return 0;                                                              \
    }

#define DELCLARE_SCC(NAME)                                                     \
    Communicator comm;                                                         \
    comm.init(argc, argv);                                                     \
    LIE NAME(grid_size, block_size);                                           \
    NAME.set_communicator(&comm);                                              \
    LIE NAME##_init(grid_size, block_size);                                    \
    NAME##_init.set_communicator(&comm);

#define EVAL_SCC(NAME)                                                         \
    KernelTimer timer;                                                         \
    timer.start_timer();                                                       \
    NAME##_init.fixpoint_loop();                                               \
    timer.stop_timer();                                                        \
    std::cout << "Init Time: " << timer.get_spent_time() << std::endl;         \
    print_memory_usage();                                                      \
    timer.start_timer();                                                       \
    NAME.fixpoint_loop();                                                      \
    timer.stop_timer();                                                        \
    std::cout << "Eval Time: " << timer.get_spent_time() << std::endl;

inline void SCC_COMPUTE(LIE &lie) {
    KernelTimer timer;
    timer.start_timer();
    lie.fixpoint_loop();
    timer.stop_timer();
    std::cout << "Eval Time: " << timer.get_spent_time() << std::endl;
}
#define SCC_INIT(NAME) SCC_COMPUTE(NAME##_init)

inline Relation *
decl_relation_input(std::string relation_name, int arity, int index,
                    std::string dataset_path,
                    std::map<column_type, std::string> &string_map,
                    int grid_size, int block_size) {
    std::stringstream relation_name_ss;
    relation_name_ss << dataset_path << "/" << relation_name << ".facts";
    thrust::host_vector<column_type> raw_relation_name_vec;
    file_to_buffer(relation_name_ss.str(), raw_relation_name_vec, string_map);
    Relation *rel = new Relation(
        relation_name, arity, raw_relation_name_vec.data(),
        raw_relation_name_vec.size() / arity, index, 0, grid_size, block_size);
    return rel;
}

#define DECLARE_RELATION_INPUT(lie, relation_name, ARITY, INDEX)               \
    Relation *relation_name =                                                  \
        decl_relation_input(#relation_name, ARITY, INDEX, dataset_path,        \
                            string_map, grid_size, block_size);                \
    lie##_init.add_relations(relation_name, true);                             \
    lie.add_relations(relation_name, true);                                    \
    relation_map[#relation_name] = relation_name;

#define DECLARE_RELATION(lie, relation_name, ARITY, INDEX)                     \
    Relation *relation_name = new Relation(#relation_name, ARITY, nullptr, 0,  \
                                           INDEX, 0, grid_size, block_size);   \
    relation_map[#relation_name] = relation_name;

#define DECLARE_RELATION_OUTPUT(lie, relation_name, ARITY, INDEX)              \
    Relation *relation_name = new Relation(#relation_name, ARITY, nullptr, 0,  \
                                           INDEX, 0, grid_size, block_size);   \
    lie.add_relations(relation_name, false);                                   \
    relation_map[#relation_name] = relation_name;

#define DECLARE_RELATION_NON_RECUR(lie, relation_name, ARITY, INDEX)           \
    Relation *relation_name = new Relation(#relation_name, ARITY, nullptr, 0,  \
                                           INDEX, 0, grid_size, block_size);   \
    lie##_init.add_relations(relation_name, true);                             \
    lie.add_relations(relation_name, false);                                   \
    relation_map[#relation_name] = relation_name;

#define DECLARE_RELATION_INPUT_OUTPUT(lie, relation_name, ARITY, INDEX)        \
    DECLARE_RELATION_INPUT(lie, relation_name, ARITY, INDEX)                   \
    lie.add_relations(relation_name, false);

#define DECLARE_TMP_RELATION(LIE, relation_name, ARITY)                        \
    Relation *relation_name = new Relation(#relation_name, ARITY, nullptr, 0,  \
                                           1, 0, grid_size, block_size);       \
    relation_map[#relation_name] = relation_name;                              \
    LIE.add_tmp_relation(relation_name);

#define INDEXED_REL(NAME, INDICES, JJC) NAME##__##INDICES##__##JJC
#define INDEXED_REL_NAME(NAME, INDICES, JJC) STRING(NAME##__##INDICES##__##JJC)

#define SYNC_INDEXED_RELATION(LIE, NAME, INDICES, JJC)                         \
    LIE.add_tail_ra(                                                           \
        RelationalCopy(NAME, NEWT, INDEXED_REL(NAME, INDICES, JJC),            \
                       TupleProjector(split_string_to_int(#INDICES)),          \
                       grid_size, block_size, true));

#define INIT_INDEXED_RELATION(LIE, NAME, INDICES, JJC)                         \
    LIE##_init.add_relations(INDEXED_REL(NAME, INDICES, JJC), false);          \
    LIE##_init.add_ra(                                                         \
        RelationalCopy(NAME, FULL, INDEXED_REL(NAME, INDICES, JJC),            \
                       TupleProjector(split_string_to_int(#INDICES)),          \
                       grid_size, block_size, false),                          \
        false);

#define CREATE_FULL_INDEXED_RELATION(LIE, NAME, ARITY, INDICES, JJC)           \
    Relation *INDEXED_REL(NAME, INDICES, JJC) =                                \
        new Relation(INDEXED_REL_NAME(NAME, INDICES, JJC), ARITY, nullptr, 0,  \
                     JJC, 0, grid_size, block_size);                           \
    relation_map[INDEXED_REL_NAME(NAME, INDICES, JJC)] =                       \
        INDEXED_REL(NAME, INDICES, JJC);                                       \
    LIE.add_relations(INDEXED_REL(NAME, INDICES, JJC), false);                 \
    INIT_INDEXED_RELATION(LIE, NAME, INDICES, JJC);                            \
    SYNC_INDEXED_RELATION(LIE, NAME, INDICES, JJC);

#define CREATE_STATIC_INDEXED_RELATION(LIE, NAME, ARITY, INDICES, JJC)         \
    Relation *INDEXED_REL(NAME, INDICES, JJC) =                                \
        new Relation(INDEXED_REL_NAME(NAME, INDICES, JJC), ARITY, nullptr, 0,  \
                     JJC, 0, grid_size, block_size);                           \
    relation_map[INDEXED_REL_NAME(NAME, INDICES, JJC)] =                       \
        INDEXED_REL(NAME, INDICES, JJC);                                       \
    LIE.add_relations(INDEXED_REL(NAME, INDICES, JJC), true);                  \
    INIT_INDEXED_RELATION(LIE, NAME, INDICES, JJC);

#define CREATE_TMP_INDEXED_RELATION(LIE, NAME, ARITY, INDICES, JJC)            \
    Relation *INDEXED_REL(NAME, INDICES, JJC) =                                \
        new Relation(INDEXED_REL_NAME(NAME, INDICES, JJC), ARITY, nullptr, 0,  \
                     JJC, 0, grid_size, block_size);                           \
    relation_map[INDEXED_REL_NAME(NAME, INDICES, JJC)] =                       \
        INDEXED_REL(NAME, INDICES, JJC);                                       \
    LIE.add_tmp_relation(INDEXED_REL(NAME, INDICES, JJC));                     \
    LIE.add_ra(RelationalSync(INDEXED_REL(NAME, INDICES, JJC), NEWT));         \
    LIE.add_ra(RelationalCopy(NAME, DELTA, INDEXED_REL(NAME, INDICES, JJC),    \
                              TupleProjector(split_string_to_int(#INDICES)),   \
                              grid_size, block_size, true));

inline void TURN_OFF_HASH(Relation *rel) { rel->index_flag = false; }

inline std::vector<int> split_string_to_int(std::string s) {
    std::vector<int> result;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, '_')) {
        result.push_back(std::stoi(token));
    }
    return result;
}

template <typename... Args> inline std::string str_concat(Args... args) {
    std::stringstream ss;
    (ss << ... << args);
    return ss.str();
}

#define IDX_REL(rel_name, indices) rel_name##_##indices

#define DECLARE_TMP_COPY(lie_name, NAME, ARITY, JJC, INDICES)                  \
    Relation *INDEXED_REL(NAME, INDICES, JJC) =                                \
        new Relation(INDEXED_REL_NAME(NAME, INDICES, JJC), ARITY, nullptr, 0,  \
                     JJC, 0, grid_size, block_size);                           \
    lie_name.add_tmp_relation(INDEXED_REL(NAME, INDICES, JJC));                \
    lie_name.add_ra(                                                           \
        RelationalCopy(NAME, DELTA, INDEXED_REL(NAME, INDICES, JJC),           \
                       TupleProjector(split_string_to_int(#INDICES)),          \
                       grid_size, block_size, true));

inline std::vector<int> generate_indices(int arity) {
    std::vector<int> result;
    for (int i = 0; i < arity; i++) {
        result.push_back(i);
    }
    return result;
}

#define DUPLICATE_TMP(lie_name, rel_name, arity, index, DUP_NAME)              \
    Relation *DUP_NAME = new Relation(#DUP_NAME, arity, nullptr, 0, index, 0,  \
                                      grid_size, block_size);                  \
    lie_name.add_tmp_relation(DUP_NAME);                                       \
    lie_name.add_ra(RelationalCopy(rel_name, NEWT, DUP_NAME,                   \
                                   TupleProjector(generate_indices(arity)),    \
                                   grid_size, block_size, true));

// #define ALIAS_RELATION(REL_NAME, REL_ALIAS) Relation *REL_ALIAS = REL_NAME;

inline std::string shashs(std::string const &str) {
    return std::to_string(s2d(str));
}

inline int n2d(int num) {
    if (num == 0) {
        return C_ZERO;
    } else {
        return -num - 16;
    }
}

inline std::string snum(int num) { return std::to_string(num); }

void MEMORY_STAT();

#define CONST_COLUMN "CONST_COLUMN"
#define JOIN_END grid_size, block_size, detail_time
#define COPY_END grid_size, block_size

// struct metavar {
//     std::string name;
// };

// using column_meta_t = std::variant<std::string, long, metavar>;
using column_meta_t = std::variant<std::string, long>;

struct clause_meta {
    std::string rel_name;
    std::vector<column_meta_t> columns;
    RelationVersion ver = FULL;
    bool negation_flag = false;

    clause_meta(std::string rel_name, std::vector<column_meta_t> columns,
                RelationVersion ver = FULL, bool negation_flag = false)
        : rel_name(rel_name), columns(columns), ver(ver),
          negation_flag(negation_flag) {}

    clause_meta(){};

    bool need_filter() {
        for (int i = 0; i < columns.size(); i++) {
            auto &col = columns[i];
            if (std::holds_alternative<long>(col)) {
                return true;
            }
            // check if 2 columns are the same
            if (std::holds_alternative<std::string>(col)) {
                if (std::get<std::string>(col).compare(WILDCARD) == 0) {
                    return true;
                }
                for (int j = i + 1; j < columns.size(); j++) {
                    if (std::holds_alternative<std::string>(columns[j])) {
                        if (std::get<std::string>(col).compare(
                                std::get<std::string>(columns[j])) == 0) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }
};

inline clause_meta negate_clause(clause_meta clause) {
    clause.negation_flag = true;
    return clause;
}

void ALIAS_RELATION(LIE &lie, LIE &lie_init, std::string rel_name,
                    std::string alias_name);

void TMP_CONST_FILTER(LIE &lie, Relation *rel,
                      std::vector<column_meta_t> target);

void TMP_CONST_COPY_FILTER(LIE &lie, Relation *dest, Relation *from,
                           RelationVersion ver, std::vector<int> indices,
                           std::vector<column_meta_t> target);

void TMP_COPY(LIE &lie, Relation *src, Relation *dst, int grid_size,
              int block_size);

void BINARY_UNION(LIE &lie, Relation *src, RelationVersion ver1, Relation dst,
                  RelationVersion ver2);

void TMP_UNION(LIE &lie, Relation *src, Relation *dst);

inline void DROP_RELATION(Relation *rel) { rel->drop(); }

// a join template function, this will compute index automatically
void BINARY_JOIN(LIE &lie, LIE &lie_init, std::string inner_name,
                 RelationVersion inner_ver, std::string outer_name,
                 RelationVersion outer_ver, std::string output_rel_name,
                 std::vector<std::string> inner_cols,
                 std::vector<std::string> outer_cols,
                 std::vector<std::string> output_cols, bool index = false,
                 bool debug = false);

void NEGATE(LIE &lie, LIE &lie_init, std::string inner_name,
            RelationVersion inner_ver, std::string outer_name,
            RelationVersion outer_ver, std::string output_name,
            std::vector<std::string> inner_cols,
            std::vector<std::string> outer_cols,
            std::vector<std::string> output_cols, bool index, bool debug);

// TODO: remove this function
void SEMI_NAIVE_BINARY_JOIN(LIE &lie, LIE &lie_init, Relation *inner_rel,
                            Relation *outer_rel, Relation *output_rel,
                            std::vector<std::string> inner_cols,
                            std::vector<std::string> outer_cols,
                            std::vector<std::string> output_cols, int grid_size,
                            int block_size, float *detail_time);

void SEMI_NAIVE_BINARY_JOIN(LIE &lie, LIE &lie_init, std::string inner_rel_name,
                            std::string outer_rel_name,
                            std::string output_rel_name,
                            std::vector<std::string> inner_cols,
                            std::vector<std::string> outer_cols,
                            std::vector<std::string> output_cols,
                            bool debug = false);

void PROJECT(LIE &lie, std::string src_rel_name, RelationVersion src_ver,
             std::string dst_rel_name, std::vector<column_meta_t> input_column,
             std::vector<column_meta_t> output_column, bool debug = false);

void FILTER_T(LIE &lie, Relation *rel, RelationVersion ver,
              std::vector<column_meta_t> target);

void FILTER_COPY(LIE &lie, std::string src_name, RelationVersion src_ver,
                 std::string dest_name, std::vector<column_meta_t> input_column,
                 std::vector<column_meta_t> output_column, bool debug = false,
                 bool tail = false);

inline void PRINT_REL_SIZE(LIE &lie, std::string name) {
    if (lie.relation_name_map.find(name) == lie.relation_name_map.end()) {
        std::cerr << "Relation not found " << name << std::endl;
    }
    Relation *rel = lie.relation_name_map[name];
    std::cout << "Relation " << name << " size: " << rel->full->tuple_counts
              << std::endl;
}

// TupleArithmetic _comp_tparith(std::vector<column_meta_t> &target,
//                               clause_meta comp) {
//     auto used_op = get_arithm_op(comp.rel_name);
//     column_meta_t lhs = comp.columns[0];
//     column_meta_t rhs = comp.columns[1];

//     std::vector<BinaryArithmeticOperator> op_list;
//     std::vector<int> lhs_pos;
//     std::vector<int> rhs_pos;

//     for (int i = 0; i < target.size(); i++) {
//         auto cur_col = target[i];
//         if (std::holds_alternative<std::string>(cur_col)) {
//             if (std::get<std::string>(cur_col).compare(
//                     std::get<std::string>(lhs)) == 0) {
//                 lhs_pos.push_back(i);
//             } else if (std::get<std::string>(cur_col).compare(
//                            std::get<std::string>(rhs)) == 0) {
//                 rhs_pos.push_back(i);
//             }
//         } else {
//             throw std::runtime_error("Invalid column type");
//         }
//     }
//     // compute lhs pos
//     return TupleArithmetic(op_list, lhs_pos, rhs_pos);
// }

void FILTER_COMP_REL(LIE &lie, std::string src_rel_name,
                     RelationVersion src_ver, std::string dest_rel_name,
                     std::vector<column_meta_t> input_column,
                     std::vector<column_meta_t> output_column,
                     clause_meta filter_comp, bool debug = false);

// void ARITH_COMP_REL(LIE &lie, std::string src_rel_nam, RelationVersion
// src_ver,
//                     std::string dest_rel_name,
//                     std::vector<column_meta_t> input_column,
//                     std::vector<column_meta_t> output_column,
//                     clause_meta arithm_comp, bool debug = false) {
//     auto src = lie.relation_name_map[src_rel_nam];
//     auto dest = lie.relation_name_map[dest_rel_name];
//     auto ra = RelationalArithmProject(
//         src, src_ver,
//     )
//     if (debug) {
//         ra.debug_flag = 0;
//     }
//     lie.add_ra(ra);

// }

// struct DatalogRules {
//   public:
//     int rule_id;
//     LIE &lie; // which lie this rule are in
//     LIE &lie_init;
//     std::vector<clause_meta> input_clauses;
//     clause_meta output_clause;

//     // constructor
//     DatalogRules(LIE &lie, LIE &lie_init, int rule_id,
//                  std::vector<clause_meta> input_clauses,
//                  clause_meta output_clause)
//         : lie(lie), lie_init(lie_init), rule_id(rule_id),
//           input_clauses(input_clauses), output_clause(output_clause) {
//         // reorder all rule, place all incremental clause at first
//         for (auto &clause : input_clauses) {
//             std::string clause_rel_name = clause.rel_name;
//             // if (!is_relname(clause_rel_name)) {
//             //     continue;
//             // }
//             if (lie.is_output_relation(clause_rel_name)) {
//                 clause.ver = DELTA;
//                 incremental_clauses.push_back(clause);
//             } else {
//                 clause.ver = FULL;
//                 non_incremental_clauses.push_back(clause);
//             }
//         }
//     }

//   private:
//     std::vector<clause_meta> incremental_clauses;
//     std::vector<clause_meta> non_incremental_clauses;
// };

// class DatalogTransProgram {
//   public:
//     LIE &lie;
//     LIE &lie_init;

//     DatalogTransProgram(LIE &lie, LIE &lie_init)
//         : lie(lie), lie_init(lie_init) {
//         grid_size = get_sm_count() * 32;
//     }

//     void add_rule(DatalogRules rule) { rules.push_back(rule); }
//     void process_rules();
//     void process_rule(DatalogRules rule);

//   private:
//     int total_rules;
//     int block_size = 512;
//     int grid_size;

//     std::vector<DatalogRules> rules;
// };

// a data log rule
// support clause:
// 1. join
// 2. project
// 3. filter
// 4. arith
// using clause_meta_t =  std::vector<column_meta_t>;
void DATALOG_RECURSIVE_RULE(LIE &lie, LIE &lie_init, int rule_id,
                            std::vector<clause_meta> input_clauses,
                            clause_meta output_clause, bool debug = false);
