
#include "../../include/exception.cuh"
#include "../../include/print.cuh"
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

    if (src->tuple_counts == 0) {
        return;
    }

    // print_tuple_rows(src, "TEST FILTER");
    // std::cout << "Before filter size : " << src->tuple_counts << std::endl;
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

void RelationalFilterProject::operator()() {
    GHashRelContainer *src;
    if (src_ver == DELTA) {
        src = src_rel->delta;
    } else if (src_ver == FULL) {
        src = src_rel->full;
    } else {
        src = src_rel->newt;
    }

    // TODO: this can only be NEWT now
    GHashRelContainer *dst;
    if (dest_ver == DELTA) {
        dst = dest_rel->delta;
    } else if (dest_ver == FULL) {
        dst = dest_rel->full;
    } else {
        dst = dest_rel->newt;
    }

    if (src->tuple_counts == 0) {
        if (debug_flag == 0) {
            std::cout << "FilterProject res cnt : " << 0 << " src is empty."
                      << std::endl;
        }
        return;
    }

    // print_tuple_rows(src, "TEST FILTER");
    // std::cout << "Before filter size : " << src->tuple_counts << std::endl;
    // count filtered
    int filtered_size =
        thrust::count_if(thrust::device, src->tuples_vec.begin(),
                         src->tuples_vec.end(), tuple_pred);

    // Allocate memory for filtered tuples
    rmm::device_vector<tuple_type> filtered_tuples_vec(filtered_size);
    thrust::copy_if(thrust::device, src->tuples_vec.begin(),
                    src->tuples_vec.end(), filtered_tuples_vec.begin(),
                    tuple_pred);

    if (debug_flag == 0) {
        std::cout << "FilterProject filtered size : " << filtered_size
                  << std::endl;
        // dump_tuple_rows(src, "value_reg", "value_reg.dump");
    }

    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount,
                           device_id);
    auto block_size = 512;
    auto grid_size = 32 * number_of_sm;

    int output_arity = dest_rel->arity;
    column_type *copied_raw_data;
    u64 copied_raw_data_size =
        filtered_size * output_arity * sizeof(column_type);
    // std::cout << "copied_raw_data_size: " << copied_raw_data_size <<
    // std::endl;
    checkCuda(cudaMalloc((void **)&copied_raw_data, copied_raw_data_size));
    checkCuda(cudaMemset(copied_raw_data, 0, copied_raw_data_size));
    get_copy_result<<<grid_size, block_size>>>(filtered_tuples_vec.data().get(),
                                               copied_raw_data, output_arity,
                                               filtered_size, tuple_generator);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    float load_relation_container_time[5] = {0, 0, 0, 0, 0};

    if (dst->tuples == nullptr || dst->tuple_counts == 0) {
        dst->free();
        load_relation_container(
            dst, dst->arity, copied_raw_data, filtered_size,
            dst->index_column_size, dst->dependent_column_size, 0.8, grid_size,
            block_size, load_relation_container_time, true, false, false);
    } else {
        GHashRelContainer *temp =
            new GHashRelContainer(dst->arity, dst->index_column_size, 0);
        temp->tuple_counts = filtered_size;
        temp->arity = output_arity;
        temp->reload(copied_raw_data, filtered_size);
        temp->sort();
        RelationalUnion ru(temp, dst);
        ru();
        dst->fit();
        temp->free();
        delete temp;
    }
    if (debug_flag == 0) {
        std::cout << "FilterProject res cnt : " << dst->tuple_counts
                  << std::endl;
        // dump_tuple_rows(dst, dest_rel->name.c_str(), "filter_project.dump");
    }
}
