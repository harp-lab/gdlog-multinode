
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include "../../include/relational_algebra.cuh"
#include "../../include/exception.cuh"
#include "../../include/print.cuh"

void RelationalArithm::operator()() {
    GHashRelContainer *src;
    if (src_ver == DELTA) {
        src = src_rel->delta;
    } else if (src_ver == FULL) {
        src = src_rel->full;
    } else {
        src = src_rel->newt;
    }

    // std::cout << "Aithmetic " << src_rel->name << std::endl;

    if (src->tuple_counts == 0) {
        return;
    }

    // transform the src by mapping TupleArithm on it
    thrust::transform(
        thrust::device,
        src->tuples,
        src->tuples + src->tuple_counts,
        src->tuples,
        tuple_generator
    );
}

void RelationalArithmProject::operator()() {
    GHashRelContainer *src;
    if (src_ver == DELTA) {
        src = src_rel->delta;
    } else if (src_ver == FULL) {
        src = src_rel->full;
    } else {
        src = src_rel->newt;
    }
    if (debug_flag == 0) {
        std::cout << "AithmProject src " << src_rel->name  << " cnt : " << src->tuple_counts << std::endl; 
        // print_tuple_rows(src, src_rel->name.c_str());
    }

    GHashRelContainer *dst;
    if (dest_ver == DELTA) {
        dst = dest_rel->delta;
    } else if (dest_ver == FULL) {
        dst = dest_rel->full;
    } else {
        dst = dest_rel->newt;
    }

    if (src->tuple_counts == 0) {
        return;
    }

    // transform the src by mapping TupleArithm on it
    thrust::transform(
        thrust::device,
        src->tuples,
        src->tuples + src->tuple_counts,
        src->tuples,
        tuple_generator
    );

    tuple_size_t arithm_tp_counts = src->tuple_counts;

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
        arithm_tp_counts * output_arity * sizeof(column_type);
    if (debug_flag == 0) {
        std::cout << "AithmProject src cnt : " << arithm_tp_counts << std::endl;
        // std::vector<column_type> test_tuple = {4, 20366, 117056};
        // tuple_generator(test_tuple.data());
        // std::cout << "AithmProject test tuple : ";
        // std::cout << tuple_generator.arity << " " << tuple_generator.left
        //           << " " << tuple_generator.right << " "
        //           << (int) tuple_generator.op << " || ";
        // for (int i = 0; i < test_tuple.size(); i++) {
        //     std::cout << test_tuple[i] << " ";
        // }
        // std::cout << std::endl;
        // print_tuple_rows(src, src_rel->name.c_str());
    }
    checkCuda(cudaMalloc((void **)&copied_raw_data, copied_raw_data_size));
    checkCuda(cudaMemset(copied_raw_data, 0, copied_raw_data_size));
    get_copy_result<<<grid_size, block_size>>>(
        src->tuples, copied_raw_data, output_arity,
        arithm_tp_counts, tuple_projector);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    float load_relation_container_time[5] = {0, 0, 0, 0, 0};

    if (dst->tuples == nullptr || dst->tuple_counts == 0) {
        dst->free();
        load_relation_container(
            dst, dst->arity, copied_raw_data, arithm_tp_counts,
            dst->index_column_size, dst->dependent_column_size, 0.8, grid_size,
            block_size, load_relation_container_time, true, false, false);
    } else {
        GHashRelContainer *temp =
            new GHashRelContainer(dst->arity, dst->index_column_size, 0);
        temp->tuple_counts = arithm_tp_counts;
        temp->arity = output_arity;
        temp->reload(copied_raw_data, arithm_tp_counts);
        temp->sort();
        RelationalUnion ru(temp, dst);
        ru();
        dst->fit();
        temp->free();
        delete temp;
    }
    if (debug_flag == 0) {
        std::cout << "AithmProject res cnt : " << dst->tuple_counts << std::endl;
        // print_tuple_rows(dst, dest_rel->name.c_str());
    }
}
