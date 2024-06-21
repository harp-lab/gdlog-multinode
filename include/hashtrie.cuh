/**
 * a GPU hash trie for relation storage.
 */

#pragma once

#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_TBB

#include "../include/relation.cuh"
#include "utils.h"
#include <cstdint>
#include <cuco/dynamic_map.cuh>
#include <cuco/static_map.cuh>
#include <cuda/std/chrono>
#include <iostream>
#include <memory>
#include <tbb/concurrent_hash_map.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <unordered_map>
#include <vector>

#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/execution_policy.h>
// #define DEFAULT_DEVICE_POLICY thrust::device
// #define DEVICE_VECTOR thrust::device_vector
#define DEFAULT_DEVICE_POLICY rmm::exec_policy()
#define DEVICE_VECTOR rmm::device_vector

namespace hisa {

using internal_data_type = uint32_t;
using device_data_t = DEVICE_VECTOR<internal_data_type>;

using offset_type = uint64_t;
// higher 32 bit is the postion in sorted indices, lower is offset
using comp_range_t = uint64_t;
using device_ranges_t = DEVICE_VECTOR<comp_range_t>;
using comp_pair_t = uint64_t;
using device_pairs_t = DEVICE_VECTOR<comp_pair_t>;

// a simple Device Map, its a wrapper of the device_vector
struct simple_map {
    device_data_t keys;
    device_ranges_t values;

    simple_map() = default;

    // bulk insert
    void insert(device_data_t &keys, device_ranges_t &values);

    // bulk find
    void find(device_data_t &keys, device_ranges_t &result);
};

// higher 32 bit is the value, lower is offset in data
// using index_value = uint64_t;
// using Map = std::unordered_map<internal_data_type, offset_type>;
using Map = tbb::concurrent_hash_map<internal_data_type, offset_type>;
using GpuMap = cuco::static_map<internal_data_type, comp_range_t>;
// using GpuMap = cuco::dynamic_map<internal_data_type, comp_range_t>;
using GpuMapPair = cuco::pair<internal_data_type, comp_range_t>;

// CSR stype column entryunique values aray in the column, sharing the same
// prefix
struct VerticalColumnGpu {

    // FIXME: remove this, this is redundant
    // all unique values in the column, sharing the same prefix
    device_data_t unique_v;
    // a mapping from the unique value to the range of tuple share the same value
    // in the next column
    std::shared_ptr<GpuMap> unique_v_map = nullptr;
    // std::unique_ptr<GpuMap> unique_v_map = nullptr;

    simple_map unique_v_map_simp;

    device_data_t sorted_indices;
    // thrust::device_vector<internal_data_type> raw_data;
    thrust::device_ptr<internal_data_type> raw_data = nullptr;

    VerticalColumnGpu() = default;

    size_t size() const { return sorted_indices.size(); }

    bool indexed = false;

    bool use_real_map = false;

    void clear_unique_v();

    ~VerticalColumnGpu();
};

struct multi_hisa {
    int arity;

    using VersionedColumns = thrust::host_vector<VerticalColumnGpu>;
    VersionedColumns full_columns;
    VersionedColumns delta_columns;
    VersionedColumns newt_columns;

    offset_type total_tuples;

    multi_hisa(int arity);

    // thrust::host_vector<int> indexed_columns;
    uint64_t hash_time = 0;
    uint64_t dedup_time = 0;
    uint64_t sort_time = 0;
    uint64_t load_time = 0;

    bool indexed = false;

    uint32_t full_size = 0;
    uint32_t delta_size = 0;
    uint32_t newt_size = 0;

    // data array, full/delta/newt all in one
    thrust::host_vector<device_data_t> data;
    // lexical order of the data in the full
    // thrust::device_vector<uint32_t> full_lexical_order;

    // load data from CPU Memory to Full, this misa must be empty
    void init_load_vectical(
        thrust::host_vector<thrust::host_vector<internal_data_type>> &tuples,
        size_t tuple_size);

    void build_index(RelationVersion version, bool sorted = false);

    // deduplicate the data in the newt
    void deduplicate();

    // load newt to delta, this will clear the newt, the delta must be empty
    // before this operation
    void new_to_delta();

    // this will
    // 1. clear the index of delta
    // 2. merge newt to full
    // 3. create the index of newt, rename to delta
    // 4. swap the newt and delta
    void persist_newt();

    void print_all(bool sorted = false);

    void fit();

    void clear();

    // get reference to the different version of columns
    VersionedColumns &get_versioned_columns(RelationVersion version) {
        switch (version) {
        case FULL:
            return full_columns;
        case DELTA:
            return delta_columns;
        case NEWT:
            return newt_columns;
        default:
            return full_columns;
        }
    }

    uint32_t get_versioned_size(RelationVersion version) {
        switch (version) {
        case FULL:
            return full_size;
        case DELTA:
            return delta_size;
        case NEWT:
            return newt_size;
        default:
            return full_size;
        }
    }

    // void deduplicate();
    uint32_t get_total_tuples() const { return total_tuples; }
};

void remove_mismatch(VerticalColumnGpu &column1, VerticalColumnGpu &column2,
                     device_data_t &match_tuple_indices,
                     device_pairs_t &matched_pair);

void column_join(VerticalColumnGpu &inner_column,
                 VerticalColumnGpu &outer_column,
                 device_data_t &outer_tuple_indices,
                 device_pairs_t &matched_indices);

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

    size_t size() const { return sorted_indices.size(); }
};

using tuple_type = thrust::host_vector<internal_data_type>;

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
