
#include "../include/exception.cuh"
#include "../include/tuple.cuh"
#include <thrust/sort.h>


__global__ void compute_hash(tuple_type *tuples, tuple_size_t rows,
                             tuple_size_t index_column_size,
                             column_type *hashes) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= rows)
        return;

    int stride = blockDim.x * gridDim.x;
    for (tuple_size_t i = index; i < rows; i += stride) {
        hashes[i] = (column_type)prefix_hash(tuples[i], index_column_size);
    }
}
