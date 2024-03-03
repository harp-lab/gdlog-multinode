#pragma once
// #include <cuda_runtime.h>
#include <functional>
#include <nvfunctional>
#include <cstdint>

using u64 = unsigned long long;
using u32 = unsigned long;

#ifndef USE_64_BIT_TUPLE
using column_type = uint32_t;
#else
using column_type = uint64_t;
#endif
using tuple_type = column_type *;
using tuple_size_t = unsigned long long;
using tuple_permutation_t = int;

#define EMPTY_HASH_ENTRY UINT64_MAX
#define MAX_ARITY 10

// TODO: use thrust vector as tuple type??
// using t_gpu_index = thrust::device_vector<u64>;
// using t_gpu_tuple = thrust::device_vector<u64>;

struct TupleGenerator {
    int reorder_map[10];
    int arity;
    int inner_arity;

    TupleGenerator(int arity, int inner_arity, std::vector<int> map) {
        this->arity = arity;
        this->inner_arity = inner_arity;
        for (int i = 0; i < arity; i++) {
            reorder_map[i] = map[i];
        }
    }

    __host__ __device__ void operator()(tuple_type inner, tuple_type outer,
                                        tuple_type result) {
        for (int i = 0; i < arity; i++) {
            if (reorder_map[i] < inner_arity) {
                result[i] = inner[reorder_map[i]];
            } else {
                result[i] = outer[reorder_map[i] - inner_arity];
            }
        }
    }
};

struct TupleProjector {
    __host__ __device__ tuple_type operator()(const tuple_type &tuple,
                                              const tuple_type &result) {
        for (int i = 0; i < arity; i++) {
            result[i] = tuple[project[i]];
        }
        return result;
    };

    int arity;
    int project[MAX_ARITY]; 

    TupleProjector(int arity, std::vector<int> project) : arity(arity) {
        for (int i = 0; i < arity; i++) {
            this->project[i] = project[i];
        }
    }
};

// using tuple_generator_hook = nvstd::function<void(tuple_type, tuple_type, tuple_type)>;

/**
 * @brief TODO: remove this use comparator function
 *
 * @param t1
 * @param t2
 * @param l
 * @return true
 * @return false
 */
__host__ __device__ inline bool tuple_eq(tuple_type t1, tuple_type t2,
                                         tuple_size_t l) {
    for (int i = 0; i < l; i++) {
        if (t1[i] != t2[i]) {
            return false;
        }
    }
    return true;
}

struct t_equal {
    u64 arity;

    t_equal(tuple_size_t arity) { this->arity = arity; }

    __host__ __device__ bool operator()(const tuple_type &lhs,
                                        const tuple_type &rhs) {
        for (int i = 0; i < arity; i++) {
            if (lhs[i] != rhs[i]) {
                return false;
            }
        }
        return true;
    }
};

/**
 * @brief fnv1-a hash used in original slog backend
 *
 * @param start_ptr
 * @param prefix_len
 * @return __host__ __device__
 */
__host__ __device__ inline column_type prefix_hash(tuple_type start_ptr,
                                                   column_type prefix_len) {
    const column_type base = 2166136261U;
    const column_type prime = 16777619U;

    column_type hash = base;
    for (column_type i = 0; i < prefix_len; ++i) {
        column_type chunk = (column_type)start_ptr[i];
        hash ^= chunk & 255U;
        hash *= prime;
        for (char j = 0; j < 3; ++j) {
            chunk = chunk >> 8;
            hash ^= chunk & 255U;
            hash *= prime;
        }
    }
    return hash;
}

// change to std
struct tuple_indexed_less {

    // u64 *index_columns;
    tuple_size_t index_column_size;
    int arity;

    tuple_indexed_less(tuple_size_t index_column_size, int arity) {
        // this->index_columns = index_columns;
        this->index_column_size = index_column_size;
        this->arity = arity;
    }

    __host__ __device__ bool operator()(const tuple_type &lhs,
                                        const tuple_type &rhs) {
        // fetch the index
        // compare hash first, could be index very different but share the same
        // hash
        // same hash
        if (lhs == 0) {
            return false;
        }
        if (rhs == 0) {
            return true;
        }
        for (tuple_size_t i = 0; i < arity; i++) {
            if (lhs[i] < rhs[i]) {
                return true;
            } else if (lhs[i] > rhs[i]) {
                return false;
            }
        }
        return false;
    }
};

struct tuple_indexed_less2 {

    // u64 *index_columns;
    tuple_size_t index_column_size;
    int arity;

    tuple_indexed_less2(tuple_size_t index_column_size, int arity) {
        // this->index_columns = index_columns;
        this->index_column_size = index_column_size;
        this->arity = arity;
    }

    __host__ __device__ bool operator()(const tuple_type &lhs,
                                        const tuple_type &rhs) {
        // fetch the index
        // compare hash first, could be index very different but share the same
        // hash
        // same hash
        if (lhs == 0) {
            return false;
        }
        if (rhs == 0) {
            return true;
        }
        if (lhs[0] < rhs[0]) {
            return true;
        } else if (lhs[0] > rhs[0]) {
            return false;
        } else {
            return lhs[1] < rhs[1];
        }
        return false;
    }
};


struct tuple_weak_less {

    int arity;

    tuple_weak_less(int arity) { this->arity = arity; }

    __host__ __device__ bool operator()(const tuple_type &lhs,
                                        const tuple_type &rhs) {

        for (u64 i = 0; i < arity; i++) {
            if (lhs[i] < rhs[i]) {
                return true;
            } else if (lhs[i] > rhs[i]) {
                return false;
            }
        }
        return false;
    };
};

__global__ void compute_hash(tuple_type *tuples, tuple_size_t rows,
                             tuple_size_t index_column_size,
                             column_type *hashes);
