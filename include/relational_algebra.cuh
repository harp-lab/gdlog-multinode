#pragma once
#include "builtin.h"
#include "relation.cuh"
#include "tuple.cuh"
#include <sstream>
#include <thrust/host_vector.h>
#include <variant>

// for fixing
#ifndef MAX_REDUCE_SIZE
#define MAX_REDUCE_SIZE 80000000
#endif

inline GHashRelContainer *get_relation_ver(Relation *rel,
                                           RelationVersion &ver) {
    if (ver == DELTA) {
        return rel->delta;
    } else if (ver == FULL) {
        return rel->full;
    } else if (ver == NEWT) {
        return rel->newt;
    } else {
        return nullptr;
    }
}

enum RAtypes {
    JOIN,
    COPY,
    ACOPY,
    FILTER,
    ARITHM,
    SYNC,
    NEGATION,
    INDEX,
    CARTESIAN,
    UNION,
    CLEAR,
    BROADCAST,
    FILTER_PROJ,
    ARITHM_PROJ
};

struct RelationalOperation {
    RAtypes type;

    int debug_flag = -1;
    virtual void print_debug_info() = 0;
};

// function hook describ how inner and outer tuple are reordered to result tuple

/**
 * @brief Relation Algerbra kernal for JOIN ⋈
 *
 */
struct RelationalJoin : public RelationalOperation {

    // relation to compare, this relation must has index
    Relation *inner_rel;
    RelationVersion inner_ver;
    // serialized relation, every tuple in this relation will be iterated and
    // joined with tuples in inner relation
    Relation *outer_rel;
    RelationVersion outer_ver;

    // the relation to store the generated join result
    Relation *output_rel;
    // hook function will be mapped on every join result tuple
    TupleGenerator tuple_generator;
    // filter to be applied on every join result tuple

    std::vector<int> reorder_map;

    int grid_size;
    int block_size;

    // flag for benchmark, this will disable sorting on result
    bool disable_load = false;

    // join time for debug and profiling
    float *detail_time;

    bool debug = false;

    TupleFilter tuple_pred;

    RelationalJoin(Relation *inner_rel, RelationVersion inner_ver,
                   Relation *outer_rel, RelationVersion outer_ver,
                   Relation *output_rel, TupleGenerator tp_gen, int grid_size,
                   int block_size, float *detail_time, bool debug = false)
        : inner_rel(inner_rel), inner_ver(inner_ver), outer_rel(outer_rel),
          outer_ver(outer_ver), output_rel(output_rel), tuple_generator(tp_gen),
          grid_size(grid_size), block_size(block_size),
          detail_time(detail_time), debug(debug) {
        if (inner_rel == nullptr || outer_rel == nullptr ||
            output_rel == nullptr) {
            throw std::runtime_error("inner, outer or output relation is null"
                                     " in RelationalJoin");
        }
        type = JOIN;
    };

    // constructor contains filter
    RelationalJoin(Relation *inner_rel, RelationVersion inner_ver,
                   Relation *outer_rel, RelationVersion outer_ver,
                   Relation *output_rel, TupleGenerator tp_gen,
                   TupleFilter tuple_pred, int grid_size, int block_size,
                   float *detail_time, bool debug = false)
        : inner_rel(inner_rel), inner_ver(inner_ver), outer_rel(outer_rel),
          outer_ver(outer_ver), output_rel(output_rel), tuple_generator(tp_gen),
          grid_size(grid_size), block_size(block_size),
          detail_time(detail_time), tuple_pred(tuple_pred), debug(debug) {
        if (inner_rel == nullptr || outer_rel == nullptr ||
            output_rel == nullptr) {
            throw std::runtime_error("inner, outer or output relation is null"
                                     " in RelationalJoin");
        }
        type = JOIN;
    };

    void operator()();

    // to string print inner ouer rel name
    void print_debug_info() {
        std::cout << "RelationalJoin << " << inner_rel->name << ", "
                  << outer_rel->name << " >>" << std::endl;
    }
};

/**
 * @brief Relation Algerbra kernal for PROJECTION Π
 *
 */
struct RelationalCopy : public RelationalOperation {
    Relation *src_rel;
    RelationVersion src_ver;
    Relation *dest_rel;
    TupleProjector tuple_generator;

    int grid_size;
    int block_size;
    bool copied = false;
    bool iterative = false;
    bool debug = false;

    RelationalCopy(Relation *src, RelationVersion src_ver, Relation *dest,
                   TupleProjector tuple_generator, int grid_size,
                   int block_size, bool iterative = false, bool debug = false)
        : src_rel(src), src_ver(src_ver), dest_rel(dest),
          tuple_generator(tuple_generator), grid_size(grid_size),
          block_size(block_size), iterative(iterative), debug(debug) {
        type = COPY;
    }

    void operator()();
    void print_debug_info() {
        std::cout << "RelationalCopy << " << src_rel->name << ", "
                  << dest_rel->name << " >>" << std::endl;
    }
};

struct RelationalFilter : public RelationalOperation {
    Relation *src_rel;
    RelationVersion src_ver;
    TupleFilter tuple_pred;

    bool copied = false;

    RelationalFilter(Relation *src, RelationVersion src_ver,
                     TupleFilter tuple_pred)
        : src_rel(src), src_ver(src_ver), tuple_pred(tuple_pred) {
        type = FILTER;
    }

    void operator()();
    void print_debug_info() {
        std::cout << "RelationalFilter << " << src_rel->name << " >>"
                  << std::endl;
    }
};

struct RelationalFilterProject : public RelationalOperation {
    Relation *src_rel;
    RelationVersion src_ver;
    TupleFilter tuple_pred;
    Relation *dest_rel;
    RelationVersion dest_ver;
    TupleProjector tuple_generator;

    bool copied = false;
    int debug_flag = -1;

    RelationalFilterProject(Relation *src, RelationVersion src_ver,
                            TupleFilter tuple_pred, Relation *dest,
                            RelationVersion dest_ver,
                            TupleProjector tuple_generator)
        : src_rel(src), src_ver(src_ver), tuple_pred(tuple_pred),
          dest_rel(dest), dest_ver(dest_ver), tuple_generator(tuple_generator) {
        if (src == nullptr) {
            throw std::runtime_error("src relation is null");
        }
        type = FILTER_PROJ;
    }

    void operator()();
    void print_debug_info() {
        std::cout << "RelationFilterProject << " << src_rel->name << " >>"
                  << std::endl;
    }
};

/**
 * @brief Relation Algerbra kernal for ARITHMETIC
 *  This operator will apply arithmetic operation on every tuple in src relation
 */
struct RelationalArithm : public RelationalOperation {
    Relation *src_rel;
    RelationVersion src_ver;
    TupleArithmetic tuple_generator;

    RelationalArithm(Relation *src, RelationVersion src_ver,
                     TupleArithmetic tuple_generator)
        : src_rel(src), src_ver(src_ver), tuple_generator(tuple_generator) {
        type = ARITHM;
    }

    void operator()();
    void print_debug_info() {
        std::cout << "RelationalArith << " << src_rel->name << " >>"
                  << std::endl;
    }
};

struct RelationalArithmProject : public RelationalOperation {
    Relation *src_rel;
    RelationVersion src_ver;
    TupleArithmeticSingle tuple_generator;
    Relation *dest_rel;
    RelationVersion dest_ver;
    TupleProjector tuple_projector;

    RelationalArithmProject(Relation *src, RelationVersion src_ver,
                            TupleArithmeticSingle tuple_generator,
                            Relation *dest, RelationVersion dest_ver,
                            TupleProjector tuple_projector)
        : src_rel(src), src_ver(src_ver), tuple_generator(tuple_generator),
          dest_rel(dest), dest_ver(dest_ver), tuple_projector(tuple_projector) {
        type = ARITHM;
    }

    void operator()();
    void print_debug_info() {
        std::cout << "RelationalArithProject << " << src_rel->name << " >>"
                  << std::endl;
    }
};

/**
 * @brief Relation Algebra kernel for sync up different indices of the same
 * relation. This RA operator must be added in the end of each SCC, it will
 * directly change the DELTA version of dest relation
 *
 */
struct RelationalACopy : public RelationalOperation {
    Relation *src_rel;
    Relation *dest_rel;
    // function will be mapped on all tuple copied
    TupleProjector tuple_generator;

    int grid_size;
    int block_size;

    RelationalACopy(Relation *src, Relation *dest,
                    TupleProjector tuple_generator, int grid_size,
                    int block_size)
        : src_rel(src), dest_rel(dest), tuple_generator(tuple_generator),
          grid_size(grid_size), block_size(block_size) {
        type = ACOPY;
    }

    void operator()();

    void print_debug_info() {
        std::cout << "RelationalACopy << " << src_rel->name << ", "
                  << dest_rel->name << " >>" << std::endl;
    }
};

// a relation algebra operator that will sync up all ranks of the same relation
// this operator will distribute the tuple to all ranks by hash of joined column
struct RelationalSync : public RelationalOperation {
    Relation *src_rel;
    RelationVersion src_ver;

    RelationalSync(Relation *src, RelationVersion src_ver)
        : src_rel(src), src_ver(src_ver) {
        type = SYNC;
    }

    void operator()() {
        // nothing happened here, the inference engine will handle the
        // communication
    };

    void print_debug_info() {
        std::cout << "RelationalSync << " << src_rel->name << " >>"
                  << std::endl;
    }
};

struct RelationalNegation : public RelationalOperation {
    Relation *src_rel;
    RelationVersion src_ver;

    Relation *neg_rel;
    RelationVersion neg_ver;

    bool left_flag = true;

    Relation *output_rel;
    TupleGenerator tuple_generator;

    int grid_size;
    int block_size;

    RelationalNegation(Relation *src, RelationVersion src_ver, Relation *neg,
                       RelationVersion neg_ver, int grid_size, int block_size)
        : src_rel(src), src_ver(src_ver), neg_rel(neg), neg_ver(neg_ver),
          grid_size(grid_size), block_size(block_size) {
        type = NEGATION;
    }

    RelationalNegation(Relation *src, RelationVersion src_ver, Relation *neg,
                       RelationVersion neg_ver, Relation *output_rel,
                       TupleGenerator tuple_generator, bool left_flag,
                       int grid_size, int block_size)
        : src_rel(src), src_ver(src_ver), neg_rel(neg), neg_ver(neg_ver),
          output_rel(output_rel), tuple_generator(tuple_generator),
          left_flag(left_flag), grid_size(grid_size), block_size(block_size) {
        type = NEGATION;
    }

    void operator()();
    void print_debug_info() {
        std::cout << "RelationalNegation << " << src_rel->name << ", "
                  << neg_rel->name << " >>" << std::endl;
    }
};

struct RelationalIndex : public RelationalOperation {
    Relation *target_rel;
    RelationVersion target_ver;

    RelationalIndex(Relation *target_rel, RelationVersion target_ver)
        : target_rel(target_rel), target_ver(target_ver) {
        type = INDEX;
    }

    void operator()() {
        // nothing happened here, the inference engine will handle the
    };

    void print_debug_info() {
        std::cout << "RelationalIndex << " << target_rel->name << " >>"
                  << std::endl;
    }
};

struct RelationalCartesian : public RelationalOperation {
    Relation *inner_rel;
    RelationVersion inner_ver;
    Relation *outer_rel;
    RelationVersion outer_ver;
    Relation *output_rel;
    TupleGenerator tuple_generator;
    TupleJoinFilter tuple_pred;

    int grid_size;
    int block_size;

    RelationalCartesian(Relation *inner_rel, RelationVersion inner_ver,
                        Relation *outer_rel, RelationVersion outer_ver,
                        Relation *output_rel, TupleGenerator tuple_generator,
                        TupleJoinFilter tuple_pred, int grid_size,
                        int block_size)
        : inner_rel(inner_rel), inner_ver(inner_ver), outer_rel(outer_rel),
          outer_ver(outer_ver), output_rel(output_rel),
          tuple_generator(tuple_generator), tuple_pred(tuple_pred),
          grid_size(grid_size), block_size(block_size) {
        type = CARTESIAN;
    }

    void operator()();
    void print_debug_info() {
        std::cout << "RelationalCartesian << " << inner_rel->name << ", "
                  << outer_rel->name << " >>" << std::endl;
    }
};

// union src to dest
struct RelationalUnion : public RelationalOperation {
    GHashRelContainer *src;
    GHashRelContainer *dest;

    RelationalUnion(GHashRelContainer *src, GHashRelContainer *dest)
        : src(src), dest(dest) {
        type = UNION;
    }

    void operator()();
    void print_debug_info() { std::cout << "RelationalUnion << " << std::endl; }
};

struct RelationalClear : public RelationalOperation {
    Relation *rel;
    RelationVersion ver;

    RelationalClear(Relation *rel, RelationVersion ver) : rel(rel), ver(ver) {
        type = CLEAR;
    }

    void operator()() { get_relation_ver(rel, ver)->free(); };

    void print_debug_info() {
        std::cout << "RelationalClear << " << rel->name << " >>" << std::endl;
    }
};

struct RelationalBroadcast : public RelationalOperation {
    GHashRelContainer *src;

    RelationalBroadcast(GHashRelContainer *src) : src(src) {}

    void operator()();
    void print_debug_info() {
        std::cout << "RelationalBroadcast << " << std::endl;
    }
};

/**
 * @brief possible RA types
 *
 */
using ra_op =
    std::variant<RelationalJoin, RelationalCopy, RelationalACopy,
                 RelationalFilter, RelationalArithm, RelationalArithmProject,
                 RelationalSync, RelationalFilterProject, RelationalNegation,
                 RelationalIndex, RelationalCartesian, RelationalUnion,
                 RelationalClear, RelationalBroadcast>;
