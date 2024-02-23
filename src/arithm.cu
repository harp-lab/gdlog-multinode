
#include "../include/relational_algebra.cuh"

#include <thrust/transform.h>
#include <thrust/execution_policy.h>

void RelationalArithm::operator()() {
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
