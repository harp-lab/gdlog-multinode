
#include "../../include/exception.cuh"
#include "../../include/relational_algebra.cuh"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/set_operations.h>



__global__ void
get_join_inner(MEntity *inner_index_map, tuple_size_t inner_index_map_size,
               tuple_size_t inner_tuple_counts, tuple_type *inner_tuples,
               tuple_type *outer_tuples, tuple_size_t outer_tuple_counts,
               int join_column_counts, bool *join_result_bitmap) {
    u64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= outer_tuple_counts)
        return;
    u64 stride = blockDim.x * gridDim.x;

    for (tuple_size_t i = index; i < outer_tuple_counts; i += stride) {
        tuple_type outer_tuple = outer_tuples[i];

        u64 hash_val = prefix_hash(outer_tuple, join_column_counts);
        // the index value "pointer" position in the index hash table
        tuple_size_t index_position = hash_val % inner_index_map_size;
        bool index_not_exists = false;
        while (true) {
            if (inner_index_map[index_position].key == hash_val &&
                tuple_eq(outer_tuple,
                         inner_tuples[inner_index_map[index_position].value],
                         join_column_counts)) {
                break;
            } else if (inner_index_map[index_position].key ==
                       EMPTY_HASH_ENTRY) {
                index_not_exists = true;
                break;
            }
            index_position = (index_position + 1) % inner_index_map_size;
        }
        if (index_not_exists) {
            continue;
        }
        // pull all joined elements
        tuple_size_t position = inner_index_map[index_position].value;
        while (true) {
            bool cmp_res = tuple_eq(inner_tuples[position], outer_tuple,
                                    join_column_counts);
            if (cmp_res) {
                join_result_bitmap[position] = true;
            } else {
                break;
            }
            position = position + 1;
            if (position > inner_tuple_counts - 1) {
                // end of data arrary
                break;
            }
        }
    }
}

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
