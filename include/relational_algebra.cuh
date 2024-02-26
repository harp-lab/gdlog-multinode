#pragma once
#include "builtin.h"
#include "relation.cuh"
#include "tuple.cuh"
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

// function hook describ how inner and outer tuple are reordered to result tuple

/**
 * @brief Relation Algerbra kernal for JOIN ⋈
 *
 */
struct RelationalJoin {

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

    TupleFilter tuple_pred;

    RelationalJoin(Relation *inner_rel, RelationVersion inner_ver,
                   Relation *outer_rel, RelationVersion outer_ver,
                   Relation *output_rel, TupleGenerator tp_gen, int grid_size,
                   int block_size, float *detail_time)
        : inner_rel(inner_rel), inner_ver(inner_ver), outer_rel(outer_rel),
          outer_ver(outer_ver), output_rel(output_rel), tuple_generator(tp_gen),
          grid_size(grid_size), block_size(block_size),
          detail_time(detail_time){};

    // constructor contains filter
    RelationalJoin(Relation *inner_rel, RelationVersion inner_ver,
                   Relation *outer_rel, RelationVersion outer_ver,
                   Relation *output_rel, TupleGenerator tp_gen,
                   TupleFilter tuple_pred, int grid_size, int block_size,
                   float *detail_time)
        : inner_rel(inner_rel), inner_ver(inner_ver), outer_rel(outer_rel),
          outer_ver(outer_ver), output_rel(output_rel), tuple_generator(tp_gen),
          grid_size(grid_size), block_size(block_size),
          detail_time(detail_time), tuple_pred(tuple_pred){};

    void operator()();
};

/**
 * @brief Relation Algerbra kernal for PROJECTION Π
 *
 */
struct RelationalCopy {
    Relation *src_rel;
    RelationVersion src_ver;
    Relation *dest_rel;
    TupleProjector tuple_generator;

    int grid_size;
    int block_size;
    bool copied = false;
    bool iterative = false;

    RelationalCopy(Relation *src, RelationVersion src_ver, Relation *dest,
                   TupleProjector tuple_generator, int grid_size,
                   int block_size, bool iterative = false)
        : src_rel(src), src_ver(src_ver), dest_rel(dest),
          tuple_generator(tuple_generator), grid_size(grid_size),
          block_size(block_size), iterative(iterative) {}

    void operator()();
};

struct RelationalFilter {
    Relation *src_rel;
    RelationVersion src_ver;
    TupleFilter tuple_pred;

    bool copied = false;

    RelationalFilter(Relation *src, RelationVersion src_ver,
                     TupleFilter tuple_pred)
        : src_rel(src), src_ver(src_ver), tuple_pred(tuple_pred) {}

    void operator()();
};

/**
 * @brief Relation Algerbra kernal for ARITHMETIC
 *  This operator will apply arithmetic operation on every tuple in src relation
 */
struct RelationalArithm {
    Relation *src_rel;
    RelationVersion src_ver;
    TupleArithmetic tuple_generator;

    RelationalArithm(Relation *src, RelationVersion src_ver,
                     TupleArithmetic tuple_generator)
        : src_rel(src), src_ver(src_ver), tuple_generator(tuple_generator) {}

    void operator()();
};

/**
 * @brief Relation Algebra kernel for sync up different indices of the same
 * relation. This RA operator must be added in the end of each SCC, it will
 * directly change the DELTA version of dest relation
 *
 */
struct RelationalACopy {
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
          grid_size(grid_size), block_size(block_size) {}

    void operator()();
};

// a relation algebra operator that will sync up all ranks of the same relation
// this operator will distribute the tuple to all ranks by hash of joined column
struct RelationalSync {
    Relation *src_rel;
    RelationVersion src_ver;

    RelationalSync(Relation *src, RelationVersion src_ver)
        : src_rel(src), src_ver(src_ver) {}

    void operator()(){
        // nothing happened here, the inference engine will handle the
        // communication
    };
};

struct RelationalNegation {
    Relation *src_rel;
    RelationVersion src_ver;

    Relation *neg_rel;
    RelationVersion neg_ver;

    int grid_size;
    int block_size;

    RelationalNegation(Relation *src, RelationVersion src_ver, Relation *neg,
                       RelationVersion neg_ver, int grid_size, int block_size)
        : src_rel(src), src_ver(src_ver), neg_rel(neg), neg_ver(neg_ver),
          grid_size(grid_size), block_size(block_size) {}

    void operator()();
};

struct RelationalIndex {
    Relation *target_rel;
    RelationVersion target_ver;

    int grid_size;
    int block_size;

    RelationalIndex(Relation *target_rel, RelationVersion target_ver,
                    int grid_size, int block_size)
        : target_rel(target_rel), target_ver(target_ver), grid_size(grid_size),
          block_size(block_size) {}

    void operator()(){
        // nothing happened here, the inference engine will handle the
    };
};

struct RelationalCartesian {
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
          grid_size(grid_size), block_size(block_size) {}

    void operator()();
};

// union src to dest
struct RelationalUnion {
    GHashRelContainer *src;
    GHashRelContainer *dest;

    RelationalUnion(GHashRelContainer *src, GHashRelContainer *dest)
        : src(src), dest(dest) {}

    void operator()();
};

struct RelationalClear {
    Relation *rel;
    RelationVersion ver;

    RelationalClear(Relation *rel, RelationVersion ver) : rel(rel), ver(ver) {}

    void operator()() { free_relation_container(get_relation_ver(rel, ver)); };
};

struct RelationalBroadcast {
    GHashRelContainer *src;

    RelationalBroadcast(GHashRelContainer *src) : src(src) {}

    void operator()();
};

/**
 * @brief possible RA types
 *
 */
using ra_op =
    std::variant<RelationalJoin, RelationalCopy, RelationalACopy,
                 RelationalFilter, RelationalArithm, RelationalSync,
                 RelationalNegation, RelationalIndex, RelationalCartesian,
                 RelationalUnion, RelationalClear, RelationalBroadcast>;

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
    BROADCAST
};
