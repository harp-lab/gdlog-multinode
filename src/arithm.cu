
#include "../include/relational_algebra.cuh"

#include <thrust/transform.h>
#include <thrust/execution_policy.h>

void RelationalArithm::operator()() {
    GHashRelContainer *src;
    if (src_ver == DELTA) {
        src = src_rel->delta;
    } else if (src_ver == FULL) {
        src = src_rel->full;
    } else {
        src = src_rel->newt;
    }

    // std::cout << "Aithmetic " << src_rel->name << std::endl;

    if (src->tuple_counts == 0) {
        return;
    }

    // transform the src by mapping TupleArithm on it
    thrust::transform(
        thrust::device,
        src->tuples,
        src->tuples + src->tuple_counts,
        src->tuples,
        tuple_generator
    );
}
