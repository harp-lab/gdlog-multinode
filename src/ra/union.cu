
#include "../../include/exception.cuh"
#include "../../include/relational_algebra.cuh"
#include "../../include/print.cuh"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <rmm/exec_policy.hpp>
#include <thrust/merge.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/pair.h>

void RelationalUnion::operator()() {
    assert(src->arity == dest->arity);
    auto arity = src->arity;

    if (src->tuple_counts <= 0) {
        return;
    }

    tuple_size_t unioned_size = src->tuple_counts + dest->tuple_counts;
    thrust::device_vector<tuple_type> merged_tuple(unioned_size);
    merged_tuple.shrink_to_fit();
    thrust::merge(
        rmm::exec_policy(), src->tuples, src->tuples + src->tuple_counts,
        dest->tuples, dest->tuples + dest->tuple_counts, merged_tuple.begin(),
        tuple_indexed_less(dest->index_column_size, dest->arity));

    column_type *merged_raw_data;
    checkCuda(cudaMalloc(&merged_raw_data, unioned_size * arity * sizeof(column_type)));
    // print_tuple_list(merged_tuple.data().get(), unioned_size, arity);

    thrust::for_each(
        thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(
            merged_tuple.begin(), thrust::counting_iterator<tuple_size_t>(0))),
        thrust::make_zip_iterator(thrust::make_tuple(
            merged_tuple.end(),
            thrust::counting_iterator<tuple_size_t>(unioned_size))),
        [dest = merged_raw_data, arity] __device__(
            const thrust::tuple<tuple_type, tuple_size_t> &t) -> void {
            auto &tuple = thrust::get<0>(t);
            auto &index = thrust::get<1>(t);
            tuple_type dest_tp = dest + index * arity;
            for (int i = 0; i < arity; i++) {
                dest_tp[i] = tuple[i];
            }
        });
    merged_tuple.clear();
    merged_tuple.shrink_to_fit();
    
    // free old data
    // free_relation_container(dest);
    dest->reload(merged_raw_data, unioned_size);
    dest->dedup();
    // print_tuple_rows(dest, "dest after merge");
}
