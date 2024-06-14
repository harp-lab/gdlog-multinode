/**
 * a GPU hash trie for relation storage.
 */

#pragma once

#include <cstdint>
#include <cuco/dynamic_map.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <unordered_map>
#include <vector>

namespace hashtrie {

using internal_data_type = uint32_t;
using offset_type = uint32_t;

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
    thrust::host_vector<internal_data_type> data;
    thrust::host_vector<uint32_t> offset;
    std::unordered_map<internal_data_type, offset_type> next;

    VerticalColumnCpu() = default;

    size_t size() const { return data.size(); }
};

struct hashtrie {
    int arity;
    std::vector<VerticalColumnGpu> columns;

    offset_type total_tuples;

    hashtrie(int arity) : arity(arity), total_tuples(0) {}

    void insert(const std::vector<internal_data_type> &tuple);

    void merge(const hashtrie &other);
};

using tuple_type = std::vector<internal_data_type>;

// cpu version of the hash trie
struct hashtrie_cpu {
    int arity;
    thrust::host_vector<VerticalColumnCpu> columns;

    offset_type total_tuples;

    hashtrie_cpu(int arity) : arity(arity), total_tuples(0) {}

    void insert(const std::vector<tuple_type> &tuples);

    void merge(const hashtrie_cpu &other);
};

} // namespace hashtrie
