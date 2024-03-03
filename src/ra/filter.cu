
#include "../../include/exception.cuh"
#include "../../include/relational_algebra.cuh"

#include <rmm/device_vector.hpp>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

void RelationalFilter::operator()() {
    GHashRelContainer *src;
    if (src_ver == DELTA) {
        src = src_rel->delta;
    } else if (src_ver == FULL) {
        src = src_rel->full;
    } else {
        src = src_rel->newt;
    }

    std::cout << "Flitering " << src_rel->name << std::endl;

    if (src->tuple_counts == 0) {
        return;
    }

    // count filtered
    int filtered_size =
        thrust::count_if(thrust::device, src->tuples_vec.begin(),
                         src->tuples_vec.end(), tuple_pred);

    // Allocate memory for filtered tuples
    rmm::device_vector<tuple_type> filtered_tuples_vec(filtered_size);
    thrust::copy_if(thrust::device, src->tuples_vec.begin(),
                    src->tuples_vec.end(), filtered_tuples_vec.begin(),
                    tuple_pred);

    // free old tuples and set new ones
    src->tuples_vec.swap(filtered_tuples_vec);
    src->tuples = src->tuples_vec.data().get();
    src->tuple_counts = filtered_size;
}
