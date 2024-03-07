#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

#include "../../include/exception.cuh"
#include "../../include/print.cuh"
#include "../../include/relational_algebra.cuh"
#include "../../include/timer.cuh"

void RelationalCopy::operator()() {
    // checkCuda(cudaDeviceSynchronize());
    GHashRelContainer *src;
    if (src_ver == DELTA) {
        src = src_rel->delta;
    } else if (src_ver == FULL) {
        src = src_rel->full;
    } else if (src_ver == NEWT) {
        src = src_rel->newt;
    } else {
        throw std::runtime_error("Invalid version");
    }
    GHashRelContainer *dest = dest_rel->newt;

    if (src->tuple_counts == 0) {
        dest_rel->newt->tuple_counts = 0;
        if (debug_flag == 1) {
            std::cout << "Copy: " << src_rel->name << " -> " << dest_rel->name
                      << " (0)" << std::endl;
        }
        return;
    }

    int output_arity = dest_rel->arity;
    column_type *copied_raw_data;
    u64 copied_raw_data_size =
        src->tuple_counts * output_arity * sizeof(column_type);
    // std::cout << "copied_raw_data_size: " << copied_raw_data_size << std::endl;
    checkCuda(cudaMalloc((void **)&copied_raw_data, copied_raw_data_size));
    checkCuda(cudaMemset(copied_raw_data, 0, copied_raw_data_size));

    auto proj_array = tuple_generator.get_project();
    // std::cout << "proj array : ";
    // for (int i = 0; i < src_rel->arity; i++) {
    //     std::cout << proj_array[i] << " ";
    // }
    // std::cout << std::endl;
    get_copy_result<<<grid_size, block_size>>>(src->tuples, copied_raw_data,
                                               output_arity, src->tuple_counts,
                                               tuple_generator);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    float load_relation_container_time[5] = {0, 0, 0, 0, 0};

    if (dest->tuples == nullptr || dest->tuple_counts == 0) {
        dest->free();
        load_relation_container(
            dest, dest->arity, copied_raw_data, src->tuple_counts,
            src->index_column_size, dest->dependent_column_size, 0.8, grid_size,
            block_size, load_relation_container_time, true, false, false);
    } else {
        GHashRelContainer *temp = new GHashRelContainer(
            dest->arity, dest->index_column_size, 0);
        temp->tuple_counts = src->tuple_counts;
        temp->arity = output_arity;
        temp->reload(copied_raw_data, src->tuple_counts);
        temp->sort();
        RelationalUnion ru(temp ,dest);
        ru();
        dest->fit();
        temp->free();
        delete temp;
    }
    // std::cout << "copied>>>>>>>>>>>> " << dest->tuple_counts << std::endl;
}
