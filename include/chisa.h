/**
 * a GPU hash trie for relation storage.
 */

#pragma once

#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_TBB

#include "../include/relation.cuh"
#include "utils.h"
#include <cstdint>
#include <iostream>
#include <memory>
#include <tbb/concurrent_hash_map.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <unordered_map>
#include <vector>

namespace hisa {

using Map = tbb::concurrent_hash_map<internal_data_type, offset_type>;

// cpu version of the vertical column
struct VerticalColumnCpu {
    // size of unique values in the column
    thrust::host_vector<internal_data_type> unique_v;
    thrust::host_vector<uint32_t> v_offset;
    // thrust::host_vector<uint32_t> v_index;
    // map from unique value to permutation of v_index
    // TODO: should this be map? or maybe normal vector? is it faster?
    Map unique_v_map;
    // size of tuples
    // sorted indices of this column
    thrust::host_vector<internal_data_type> sorted_indices;
    thrust::host_vector<internal_data_type> raw_data;

    VerticalColumnCpu() = default;

    size_t size() const { return sorted_indices.size(); }
};

// cpu version of the hash trie
struct hisa_cpu {
    int arity;
    thrust::host_vector<VerticalColumnCpu> columns;

    // thrust::host_vector<int> indexed_columns;
    uint64_t hash_time = 0;

    bool indexed = false;

    offset_type total_tuples;

    hisa_cpu(int arity) : arity(arity), total_tuples(0) {}

    void fetch_column_values(int column,
                             thrust::host_vector<internal_data_type> &values,
                             bool sorted = false);

    void
    fetch_column_unique_values(int column,
                               thrust::host_vector<internal_data_type> &values);

    void load(const thrust::host_vector<tuple_type> &tuples);
    void load_vectical(
        thrust::host_vector<thrust::host_vector<internal_data_type>> &tuples);

    // merge will move out the undupilcaate  data from other
    void merge(hisa_cpu &other);

    void remove_dup_in(hisa_cpu &other);

    void build_index(bool sorted = false);

    void deduplicate();

    void column_join(int column, hisa_cpu &other, int other_column,
                     thrust::host_vector<uint32_t> &result);

    void print_all(bool sorted = false);

    void clear();

    // void deduplicate();
    uint32_t get_total_tuples() const { return total_tuples; }
};

} // namespace hisa
