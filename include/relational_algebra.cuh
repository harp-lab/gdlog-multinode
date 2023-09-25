#pragma once
#include "relation.cuh"
#include "tuple.cuh"
#include <thrust/host_vector.h>
#include <variant>

// function hook describ how inner and outer tuple are reordered to result tuple

struct RelationalJoin {

    Relation *inner_rel;
    RelationVersion inner_ver;
    Relation *outer_rel;
    RelationVersion outer_ver;

    Relation *output_rel;
    tuple_generator_hook tuple_generator;
    tuple_predicate tuple_pred;

    JoinDirection direction;
    int grid_size;
    int block_size;
    float *detail_time;

    RelationalJoin(Relation *inner_rel, RelationVersion inner_ver,
                   Relation *outer_rel, RelationVersion outer_ver,
                   Relation *output_rel, tuple_generator_hook tp_gen,
                   tuple_predicate tp_pred, JoinDirection direction,
                   int grid_size, int block_size, float *detail_time)
        : inner_rel(inner_rel), inner_ver(inner_ver), outer_rel(outer_rel),
          outer_ver(outer_ver), output_rel(output_rel), tuple_generator(tp_gen),
          tuple_pred(tp_pred), direction(direction), grid_size(grid_size),
          block_size(block_size), detail_time(detail_time){};

    void operator()();
};

struct RelationalCopy {
    Relation *src_rel;
    RelationVersion src_ver;
    Relation *dest_rel;
    tuple_copy_hook tuple_generator;
    tuple_predicate tuple_pred;

    int grid_size;
    int block_size;
    bool copied = false;

    RelationalCopy(Relation *src, RelationVersion src_ver, Relation *dest,
                   tuple_copy_hook tuple_generator, tuple_predicate tuple_pred,
                   int grid_size, int block_size)
        : src_rel(src), src_ver(src_ver), dest_rel(dest),
          tuple_generator(tuple_generator), tuple_pred(tuple_pred),
          grid_size(grid_size), block_size(block_size) {}

    void operator()();
};

struct RelationalACopy {
    Relation *src_rel;
    Relation *dest_rel;
    tuple_copy_hook tuple_generator;
    tuple_predicate tuple_pred;

    int grid_size;
    int block_size;

    RelationalACopy(Relation *src, Relation *dest,
                    tuple_copy_hook tuple_generator, tuple_predicate tuple_pred,
                    int grid_size, int block_size)
        : src_rel(src), dest_rel(dest), tuple_generator(tuple_generator),
          tuple_pred(tuple_pred), grid_size(grid_size), block_size(block_size) {
    }

    void operator()();
};

using ra_op = std::variant<RelationalJoin, RelationalCopy, RelationalACopy>;

enum RAtypes { JOIN, COPY, ACOPY };