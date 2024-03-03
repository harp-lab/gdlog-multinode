#include <iostream>
#include <mpi.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/unique.h>
#include <vector>
#include <rmm/exec_policy.hpp>

#include "../../include/exception.cuh"
#include "../../include/print.cuh"
#include "../../include/relational_algebra.cuh"
#include "../../include/timer.cuh"


__global__ void get_join_result_size(GHashRelContainer *inner_table,
                                     GHashRelContainer *outer_table,
                                     int join_column_counts,
                                     TupleGenerator tp_gen, TupleFilter tp_pred,
                                     tuple_size_t *join_result_size) {
    u64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= outer_table->tuple_counts)
        return;
    u64 stride = blockDim.x * gridDim.x;

    for (tuple_size_t i = index; i < outer_table->tuple_counts; i += stride) {
        tuple_type outer_tuple = outer_table->tuples[i];

        tuple_size_t current_size = 0;
        join_result_size[i] = 0;
        u64 hash_val = prefix_hash(outer_tuple, outer_table->index_column_size);
        // the index value "pointer" position in the index hash table
        tuple_size_t index_position = hash_val % inner_table->index_map_size;
        bool index_not_exists = false;
        while (true) {
            if (inner_table->index_map[index_position].key == hash_val &&
                tuple_eq(
                    outer_tuple,
                    inner_table
                        ->tuples[inner_table->index_map[index_position].value],
                    outer_table->index_column_size)) {
                break;
            } else if (inner_table->index_map[index_position].key ==
                       EMPTY_HASH_ENTRY) {
                index_not_exists = true;
                break;
            }
            index_position = (index_position + 1) % inner_table->index_map_size;
        }
        if (index_not_exists) {
            continue;
        }
        // pull all joined elements
        tuple_size_t position = inner_table->index_map[index_position].value;
        while (true) {
            tuple_type cur_inner_tuple = inner_table->tuples[position];
            bool cmp_res = tuple_eq(inner_table->tuples[position], outer_tuple,
                                    join_column_counts);
            if (cmp_res) {
                // hack to apply filter
                // TODO: this will cause max arity of a relation is 20
                if (tp_pred.arity > 0) {
                    column_type tmp[10] = {0};
                    tp_gen(cur_inner_tuple, outer_tuple, tmp);
                    if (tp_pred(tmp)) {
                        current_size++;
                    }
                } else {
                    current_size++;
                }
            } else {
                break;
            }
            position = position + 1;
            if (position > inner_table->tuple_counts - 1) {
                // end of data arrary
                break;
            }
        }
        join_result_size[i] = current_size;
    }
}

__global__ void
get_join_result(GHashRelContainer *inner_table, GHashRelContainer *outer_table,
                int join_column_counts, TupleGenerator tp_gen,
                TupleFilter tp_pred, int output_arity,
                column_type *output_raw_data, tuple_size_t *res_count_array,
                tuple_size_t *res_offset, JoinDirection direction) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= outer_table->tuple_counts)
        return;

    int stride = blockDim.x * gridDim.x;

    for (tuple_size_t i = index; i < outer_table->tuple_counts; i += stride) {
        if (res_count_array[i] == 0) {
            continue;
        }
        tuple_type outer_tuple = outer_table->tuples[i];

        int current_new_tuple_cnt = 0;
        u64 hash_val = prefix_hash(outer_tuple, outer_table->index_column_size);
        // the index value "pointer" position in the index hash table
        tuple_size_t index_position = hash_val % inner_table->index_map_size;
        bool index_not_exists = false;
        while (true) {
            if (inner_table->index_map[index_position].key == hash_val &&
                tuple_eq(
                    outer_tuple,
                    inner_table
                        ->tuples[inner_table->index_map[index_position].value],
                    outer_table->index_column_size)) {
                break;
            } else if (inner_table->index_map[index_position].key ==
                       EMPTY_HASH_ENTRY) {
                index_not_exists = true;
                break;
            }
            index_position = (index_position + 1) % inner_table->index_map_size;
        }
        if (index_not_exists) {
            continue;
        }

        // pull all joined elements
        tuple_size_t position = inner_table->index_map[index_position].value;
        while (true) {
            // TODO: always put join columns ahead? could be various benefits
            // but memory is issue to mantain multiple copies
            bool cmp_res = tuple_eq(inner_table->tuples[position], outer_tuple,
                                    join_column_counts);
            if (cmp_res) {
                // tuple prefix match, join here
                tuple_type inner_tuple = inner_table->tuples[position];
                tuple_type new_tuple =
                    output_raw_data +
                    (res_offset[i] + current_new_tuple_cnt) * output_arity;

                // for (int j = 0; j < output_arity; j++) {
                // TODO: this will cause max arity of a relation is 20
                if (tp_pred.arity > 0) {
                    column_type tmp[20];
                    tp_gen(inner_tuple, outer_tuple, tmp);
                    if (tp_pred(tmp)) {
                        tp_gen(inner_tuple, outer_tuple, new_tuple);
                        current_new_tuple_cnt++;
                    }
                } else {
                    tp_gen(inner_tuple, outer_tuple, new_tuple);
                    current_new_tuple_cnt++;
                }
                if (current_new_tuple_cnt > res_count_array[i]) {
                    break;
                }
            } else {
                // bucket end
                break;
            }
            position = position + 1;
            if (position > (inner_table->tuple_counts - 1)) {
                // end of data arrary
                break;
            }
        }
    }
}

void RelationalJoin::operator()() {

    bool output_is_tmp = output_rel->tmp_flag;
    GHashRelContainer *inner;
    if (inner_ver == DELTA) {
        inner = inner_rel->delta;
    } else {
        inner = inner_rel->full;
    }
    GHashRelContainer *outer;
    if (outer_ver == DELTA) {
        outer = outer_rel->delta;
    } else if (outer_ver == FULL) {
        outer = outer_rel->full;
    } else {
        // temp relation can be outer relation
        outer = outer_rel->newt;
    }
    int output_arity = output_rel->arity;
    // GHashRelContainer* output = output_rel->newt;

    if (outer->tuples == nullptr || outer->tuple_counts == 0) {
        outer->tuple_counts = 0;
        return;
    }
    if (inner->tuples == nullptr || inner->tuple_counts == 0) {
        outer->tuple_counts = 0;
        return;
    }

    KernelTimer timer;
    // checkCuda(cudaDeviceSynchronize());
    GHashRelContainer *inner_device;
    checkCuda(cudaMalloc((void **)&inner_device, sizeof(GHashRelContainer)));
    checkCuda(cudaMemcpy(inner_device, inner, sizeof(GHashRelContainer),
                         cudaMemcpyHostToDevice));
    GHashRelContainer *outer_device;
    checkCuda(cudaMalloc((void **)&outer_device, sizeof(GHashRelContainer)));
    checkCuda(cudaMemcpy(outer_device, outer, sizeof(GHashRelContainer),
                         cudaMemcpyHostToDevice));

    tuple_size_t *result_counts_array;
    checkCuda(cudaMalloc((void **)&result_counts_array,
                         outer->tuple_counts * sizeof(tuple_size_t)));
    checkCuda(cudaMemset(result_counts_array, 0,
                         outer->tuple_counts * sizeof(tuple_size_t)));

    // checkCuda(cudaDeviceSynchronize());
    timer.start_timer();
    checkCuda(cudaDeviceSynchronize());
    get_join_result_size<<<grid_size, block_size>>>(
        inner_device, outer_device, outer->index_column_size, tuple_generator,
        tuple_pred, result_counts_array);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    this->detail_time[0] += timer.get_spent_time();

    timer.start_timer();
    tuple_size_t total_result_rows = 0;
    for (tuple_size_t i = 0; i < outer->tuple_counts; i = i + MAX_REDUCE_SIZE) {
        tuple_size_t reduce_size = MAX_REDUCE_SIZE;
        if (i + MAX_REDUCE_SIZE > outer->tuple_counts) {
            reduce_size = outer->tuple_counts - i;
        }
        tuple_size_t reduce_v =
            thrust::reduce(rmm::exec_policy(), result_counts_array + i,
                           result_counts_array + i + reduce_size, 0);
        total_result_rows += reduce_v;
        // checkCuda(cudaDeviceSynchronize());
    }

    tuple_size_t *result_counts_offset;
    checkCuda(cudaMalloc((void **)&result_counts_offset,
                         outer->tuple_counts * sizeof(tuple_size_t)));
    checkCuda(cudaMemcpy(result_counts_offset, result_counts_array,
                         outer->tuple_counts * sizeof(tuple_size_t),
                         cudaMemcpyDeviceToDevice));
    thrust::exclusive_scan(rmm::exec_policy(), result_counts_offset,
                           result_counts_offset + outer->tuple_counts,
                           result_counts_offset);

    timer.stop_timer();
    detail_time[1] += timer.get_spent_time();

    timer.start_timer();
    column_type *join_res_raw_data;
    u64 join_res_raw_data_mem_size =
        total_result_rows * output_arity * sizeof(column_type);
    checkCuda(
        cudaMalloc((void **)&join_res_raw_data, join_res_raw_data_mem_size));
    checkCuda(cudaMemset(join_res_raw_data, 0, join_res_raw_data_mem_size));
    get_join_result<<<grid_size, block_size>>>(
        inner_device, outer_device, outer->index_column_size, tuple_generator,
        tuple_pred, output_arity, join_res_raw_data, result_counts_array,
        result_counts_offset, LEFT);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    detail_time[2] += timer.get_spent_time();
    checkCuda(cudaFree(result_counts_array));
    checkCuda(cudaFree(result_counts_offset));

    float load_relation_container_time[5] = {0, 0, 0, 0, 0};
    // // reload newt
    // free_relation(output_newt);
    // newt don't need index
    if (output_rel->newt->tuples == nullptr ||
        output_rel->newt->tuple_counts == 0) {
        if (disable_load) {
            return;
        }
        if (!output_is_tmp) {
            load_relation_container(
                output_rel->newt, output_arity, join_res_raw_data,
                total_result_rows, output_rel->index_column_size,
                output_rel->dependent_column_size, 0.8, grid_size, block_size,
                load_relation_container_time, true, false, false);
        } else {
            // temporary relation doesn't need index nor sort
            load_relation_container(
                output_rel->newt, output_arity, join_res_raw_data,
                total_result_rows, output_rel->index_column_size,
                output_rel->dependent_column_size, 0.8, grid_size, block_size,
                load_relation_container_time, true, true, false);
            output_rel->newt->tmp_flag = true;
        }
        checkCuda(cudaDeviceSynchronize());
        detail_time[3] += load_relation_container_time[0];
        detail_time[4] += load_relation_container_time[1];
        detail_time[5] += load_relation_container_time[2];
    } else {
        // TODO: handle the case out put relation is temp relation
        // data in current newt, merge
        if (!output_is_tmp) {
            GHashRelContainer *newt_tmp = new GHashRelContainer(
                output_rel->arity, output_rel->index_column_size,
                output_rel->dependent_column_size);
            GHashRelContainer *old_newt = output_rel->newt;
            load_relation_container(
                newt_tmp, output_arity, join_res_raw_data, total_result_rows,
                output_rel->index_column_size,
                output_rel->dependent_column_size, 0.8, grid_size, block_size,
                load_relation_container_time, true, false, false);
            detail_time[3] += load_relation_container_time[0];
            detail_time[4] += load_relation_container_time[1];
            detail_time[5] += load_relation_container_time[2];
            RelationalUnion ru(newt_tmp, output_rel->newt);
            ru();
            output_rel->newt->fit();
            newt_tmp->free();
            delete newt_tmp;
        } else {
            // output relation is tmp relation, directly merge without sort
            GHashRelContainer *old_newt = output_rel->newt;
            column_type *newt_tmp_raw;
            u64 newt_tmp_raw_mem_size =
                (old_newt->tuple_counts + total_result_rows) *
                output_rel->arity * sizeof(column_type);
            tuple_size_t new_newt_counts =
                old_newt->tuple_counts + total_result_rows;
            checkCuda(
                cudaMalloc((void **)&newt_tmp_raw, newt_tmp_raw_mem_size));
            checkCuda(cudaMemcpy(newt_tmp_raw, old_newt->data_raw,
                                 old_newt->tuple_counts * old_newt->arity *
                                     sizeof(column_type),
                                 cudaMemcpyDeviceToDevice));
            checkCuda(cudaMemcpy(
                &(newt_tmp_raw[old_newt->tuple_counts * old_newt->arity]),
                join_res_raw_data,
                total_result_rows * output_rel->arity * sizeof(column_type),
                cudaMemcpyDeviceToDevice));
            old_newt->free();
            checkCuda(cudaFree(join_res_raw_data));
            load_relation_container(
                output_rel->newt, output_arity, newt_tmp_raw, new_newt_counts,
                output_rel->index_column_size,
                output_rel->dependent_column_size, 0.8, grid_size, block_size,
                load_relation_container_time, true, true, false);
            checkCuda(cudaDeviceSynchronize())
        }

        detail_time[3] += load_relation_container_time[0];
        detail_time[4] += load_relation_container_time[1];
        detail_time[5] += load_relation_container_time[2];
        // print_tuple_rows(output_rel->newt, "join merge newt");
        // delete newt_tmp;
    }

    // print_tuple_rows(output_rel->newt, "output_newtr");
    // checkCuda(cudaDeviceSynchronize());
    // std::cout << output_rel->name << " join result size " <<
    // output_rel->newt->tuple_counts <<std::endl;

    checkCuda(cudaFree(inner_device));
    checkCuda(cudaFree(outer_device));
}
