
#include "../include/relational_algebra.cuh"
#include "../include/exception.cuh"

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
// #include <rmm/device_vector.h>

void RelationalFilter::operator()() {
    
    std::cout << "Flitering " << src_rel->name << std::endl;

    if (src->tuple_counts == 0) {
        return;
    }

    // count filtered
    int filtered_size = thrust::count_if(
        thrust::device, src->tuples, src->tuples + src->tuple_counts, tuple_pred);

    // Allocate memory for filtered tuples
    tuple_type *filtered_tuples;
    checkCuda(cudaMalloc(&filtered_tuples, filtered_size * sizeof(tuple_type)));
    thrust::copy_if(
        thrust::device, src->tuples, src->tuples + src->tuple_counts,
        filtered_tuples, tuple_pred);
    
    // free old tuples and set new ones
    checkCuda(cudaFree(src->tuples));
    src->tuples = filtered_tuples;
    src->tuple_counts = filtered_size;
}
