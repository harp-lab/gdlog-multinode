/**
 * a GPU hash trie for relation storage.
 */

#pragma once

#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_TBB

#include <cstdint>
#include <cuco/dynamic_map.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <unordered_map>
#include <vector>
#include <tbb/concurrent_hash_map.h>
#include <cuda/std/chrono>

namespace hisa {

using internal_data_type = uint32_t;
using offset_type = uint32_t;
// using Map = std::unordered_map<internal_data_type, offset_type>;
using Map = tbb::concurrent_hash_map<internal_data_type, offset_type>;

// CSR stype column entryunique values aray in the column, sharing the same
// prefix
struct VerticalColumnGpu {
    // all unique values in the column, sharing the same prefix
    thrust::device_vector<internal_data_type> data;
    // for each unique value, how many tuples have this value
    thrust::device_vector<uint32_t> offset;
    // a mapping from the unique value to the index of the first tuple
    // in the next column
    cuco::dynamic_map<internal_data_type, offset_type> next;

    VerticalColumnGpu() = default;

    size_t size() const { return data.size(); }
};

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

    // check
    // https://oneapi-src.github.io/oneTBB/main/tbb_userguide/concurrent_hash_map.html

    VerticalColumnCpu() = default;

    size_t size() const { return raw_data.size(); }
};

struct multi_hisa {
    int arity;
    std::vector<VerticalColumnGpu> columns;

    offset_type total_tuples;

    multi_hisa(int arity) : arity(arity), total_tuples(0) {}

    void load(const std::vector<internal_data_type> &tuple);

    void merge(const multi_hisa &other);
};

using tuple_type = std::vector<internal_data_type>;

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
    
    void fetch_column_unique_values(int column,
                                    thrust::host_vector<internal_data_type> &values);

    void load(const std::vector<tuple_type> &tuples);
    void load_vectical(std::vector<std::vector<internal_data_type>> &tuples);

    // merge will move out the undupilcaate  data from other
    void merge(hisa_cpu &other);

    void remove_dup_in(hisa_cpu &other);

    void build_index(bool sorted = false);

    void deduplicate();

    void column_join(int column, hisa_cpu &other, int other_column,
                     std::vector<uint32_t> &result);

    void print_all(bool sorted = false);

    void clear();

    // void deduplicate();
    uint32_t get_total_tuples() const { return total_tuples; }
};

} // namespace hashtrie
