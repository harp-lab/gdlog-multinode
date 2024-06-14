
#include "../include/hashtrie.cuh"

#include <thrust/adjacent_difference.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace hashtrie {

// hashtrie::insert(const std::vector<internal_data_type> &tuple) {

// }

void hashtrie_cpu::insert(const std::vector<tuple_type> &tuples) {

    thrust::host_vector<internal_data_type> indices(tuples.size());
    thrust::sequence(indices.begin(), indices.end());

    for (int i = 0; i < arity - 1; i++) {
        // extract the i-th column
        thrust::host_vector<internal_data_type> column_data(tuples.size());
        thrust::transform(indices.begin(), indices.end(),
                          tuples.begin(), column_data.begin(),
                          [i](auto j, auto &tuple) { return tuple[i]; });
        // extract the i+1-th column
        thrust::host_vector<internal_data_type> next_column_data(tuples.size());
        thrust::transform(indices.begin(), indices.end(),
                          tuples.begin(), next_column_data.begin(),
                          [i](auto j, auto &tuple) { return tuple[i + 1]; });
        // sort all values in the column
        thrust::stable_sort_by_key(column_data.begin(), column_data.end(), indices.begin());
        // compress the column, save unique values and their counts
        thrust::host_vector<internal_data_type> unique_data(column_data.size());
        thrust::host_vector<uint32_t> unique_offset(column_data.size());
        thrust::sequence(unique_offset.begin(), unique_offset.end());
        // using thrust parallel algorithm to compress the column
        // use scan
        // mark non-unique values as 0
        thrust::unique_by_key(column_data.begin(), column_data.end(),
                              unique_offset.begin());
        // calculate offset by minus the previous value
        thrust::adjacent_difference(unique_offset.begin(), unique_offset.end(),
                                    unique_offset.begin());
        
    }

}

} // namespace hashtrie

