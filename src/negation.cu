
#include "../include/exception.cuh"
#include "../include/relational_algebra.cuh"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/set_operations.h>

void RelationalNegation::operator()() {
    GHashRelContainer *src;
    if (src_ver == DELTA) {
        src = src_rel->delta;
    } else if (src_ver == FULL) {
        src = src_rel->full;
    } else {
        src = src_rel->newt;
    }
    GHashRelContainer *negate;
    if (neg_ver == DELTA) {
        negate = neg_rel->delta;
    } else if (neg_ver == FULL) {
        negate = neg_rel->full;
    } else {
        negate = neg_rel->newt;
    }

    int jcc = neg_rel->index_column_size;

    thrust::device_vector<bool> result_bitmap_vec(src->tuple_counts, false);
    get_join_inner<<<grid_size, block_size>>>(
        src->index_map, src->index_map_size, src->tuple_counts, src->tuples,
        negate->tuples, negate->tuple_counts, jcc,
        result_bitmap_vec.data().get());
    
    std::cout << "Negation result bitmap: ";
    for (int i = 0; i < src->tuple_counts; i++) {
        std::cout << result_bitmap_vec[i] << " ";
    }

    auto new_tuple_end = thrust::remove_if(
        thrust::device, src->tuples, src->tuples + src->tuple_counts,
        result_bitmap_vec.begin(), thrust::identity<bool>());

    src->tuple_counts = new_tuple_end - src->tuples;
}
