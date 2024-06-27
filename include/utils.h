
#pragma once

#include <cstdint>
#include <thrust/host_vector.h>

enum RelationVersion { DELTA, FULL, NEWT };

namespace hisa {
    using internal_data_type = uint32_t;

    using offset_type = uint64_t;
    using comp_range_t = uint64_t;
    using comp_pair_t = uint64_t;

    using tuple_type = thrust::host_vector<internal_data_type>;

    inline uint64_t __device__ __host__ compress_u32(uint32_t &a, uint32_t &b) {
    return ((uint64_t)a << 32) | b;
}
}
