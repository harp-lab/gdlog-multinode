#include "../include/dynamic_dispatch.h"
#include "../include/exception.cuh"
#include "../include/lie.cuh"
#include "../include/print.cuh"
#include "../include/timer.cuh"
#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>

#include <variant>

void LIE::add_ra(ra_op op) { ra_ops.push_back(op); }

void LIE::add_relations(Relation *rel, bool static_flag) {
    if (static_flag) {
        static_relations.push_back(rel);
    } else {
        update_relations.push_back(rel);
        // add delta and newt for it
    }
}

void LIE::add_tmp_relation(Relation *rel) { tmp_relations.push_back(rel); }

void LIE::redistribute_full_relations() {
    if (mcomm->isInitialized()) {
        for (Relation *rel : update_relations) {
            mcomm->distribute_bucket(rel->fulls[0]);
            rel->fulls[0]->build_index(grid_size, block_size);
        }
        for (Relation *rel : static_relations) {
            mcomm->distribute_bucket(rel->fulls[0]);
            rel->fulls[0]->build_index(grid_size, block_size);
        }
    }
}

GHashRelContainer *get_relation_container(Relation *rel, RelationVersion ver,
                                          bucket_id_t bucket_id) {
    if (ver == FULL) {
        return rel->fulls[bucket_id];
    } else if (ver == DELTA) {
        return rel->deltas[bucket_id];
    } else {
        return rel->newts[bucket_id];
    }
}

void LIE::fixpoint_loop() {

    int iteration_counter = 0;
    float join_time = 0;
    float merge_time = 0;
    float rebuild_time = 0;
    float flatten_time = 0;
    float set_diff_time = 0;
    float rebuild_delta_time = 0;
    float flatten_full_time = 0;
    float memory_alloc_time = 0;

    float join_get_size_time = 0;
    float join_get_result_time = 0;
    float rebuild_newt_time = 0;
    KernelTimer timer;

    float rebuild_rel_sort_time = 0;
    float rebuild_rel_unique_time = 0;
    float rebuild_rel_index_time = 0;

    float node_comm_time = 0;

    if (mcomm->isInitialized()) {
        if (mcomm->getRank() == 0) {
            std::cout << "start lie .... " << std::endl;
        }
        timer.start_timer();
        redistribute_full_relations();
        timer.stop_timer();
        node_comm_time += timer.get_spent_time();
    } else {
        std::cout << "start lie .... " << std::endl;
    }
    // init full tuple buffer for all relation involved
    for (Relation *rel : update_relations) {
        for (int b_id = 0; b_id < rel->sub_bucket_size; b_id++) {
            auto full = rel->fulls[b_id];
            auto delta = rel->deltas[b_id];
            auto tuple_full = rel->tuple_fulls[b_id];
//            auto current_full_size = rel->current_full_sizes[b_id];
            checkCuda(cudaMalloc((void **)&tuple_full,
                                 full->tuple_counts *
                                     sizeof(tuple_type)));
            checkCuda(
                cudaMemcpy(tuple_full, full->tuples,
                           full->tuple_counts * sizeof(tuple_type),
                           cudaMemcpyDeviceToDevice));
            rel->current_full_sizes[b_id] = full->tuple_counts;
            copy_relation_container(delta, full, grid_size, block_size);
            checkCuda(cudaDeviceSynchronize());
        }
    }
    while (true) {
        auto prev_comm_time = node_comm_time;
        for (auto &ra_op : ra_ops) {
            timer.start_timer();
            std::visit(
                dynamic_dispatch{
                    [&](RelationalJoin &op) {
                        // timer.start_timer();
                        op.inner = get_relation_container(op.inner_rel,
                                                          op.inner_ver, 0);
                        op.output = op.output_rel->newts[0];
                        op();
                    },
                    [&](RelationalACopy &op) {
                        op.src = op.src_rel->newts[0];
                        op.dest = op.dest_rel->newts[0];
                        op();
                    },
                    [&](RelationalCopy &op) {
                        op.src = op.src_rel->newts[0];
                        op.dest = op.dest_rel->newts[0];
                        if (op.src->tuple_counts == 0) {
                            op.dest->tuple_counts = 0;
                        }
                        if (op.src_ver == FULL) {
                            if (!op.copied) {
                                op();
                                op.copied = true;
                            }
                        } else {
                            op();
                        }
                    },
                    [&](RelationalFilter &op) {
                        op.src =
                            get_relation_container(op.src_rel, op.src_ver, 0);
                        op();
                    },
                    [&](RelationalArithm &op) {
                        op.src =
                            get_relation_container(op.src_rel, op.src_ver, 0);
                        op();
                    },
                    [&](RelationalBucketSync &op) {
                        if (op.src_ver == FULL) {
                            mcomm->distribute_bucket(op.src_rel->fulls[0]);
                        } else if (op.src_ver == DELTA) {
                            mcomm->distribute_bucket(op.src_rel->deltas[0]);
                        } else {
                            // std::cout << ">>>>>>>>>>>>>>>>>>
                            // sync " << mcomm->getTotalRank() <<
                            // std::endl;
                            mcomm->distribute_bucket(op.src_rel->newts[0]);
                        }
                    }},
                ra_op);
            timer.stop_timer();
            join_time += timer.get_spent_time();
        }

        // clean tmp relation
        for (Relation *rel : tmp_relations) {
            for (int b_id = 0; b_id < rel->sub_bucket_size; b_id++) {
                free_relation_container(rel->newts[b_id]);
            }
        }

        // merge delta into full
        bool fixpoint_flag = true;
        for (Relation *rel : update_relations) {

            if (iteration_counter == 0) {
                for (int b_id = 0; b_id < rel->sub_bucket_size; b_id++) {
                    free_relation_container(rel->deltas[b_id]);
                }
            }

            // drop the index of delta once merged, because it won't be used in
            // next iter when migrate more general case, this operation need to
            // be put off to end of all RA operation in current iteration
            for (int b_id = 0; b_id < rel->sub_bucket_size; b_id++) {
                auto delta = rel->deltas[b_id];
                if (delta->index_map != nullptr) {
                    checkCuda(cudaFree(delta->index_map));
                    delta->index_map = nullptr;
                }
                if (delta->tuples != nullptr) {
                    checkCuda(cudaFree(delta->tuples));
                    delta->tuples = nullptr;
                }
            }

            if (mcomm->isInitialized()) {
                // mutil GPU, distributed newt
                timer.start_timer();
                mcomm->distribute_bucket(rel->newts[0]);
                timer.stop_timer();
                node_comm_time += timer.get_spent_time();
                // gather newt size
            }

            for (int b_id = 0; b_id < rel->sub_bucket_size; b_id++) {
                auto newt = rel->newts[b_id];
                auto full = rel->fulls[b_id];
                auto delta = rel->deltas[b_id];
                auto tuple_full = rel->tuple_fulls[b_id];
                auto current_full_size = rel->current_full_sizes[b_id];
                tuple_size_t newt_size = newt->tuple_counts;
                timer.start_timer();
                tuple_type *deduplicated_newt_tuples;
                tuple_size_t deduplicate_size = 0;
                if (newt_size != 0) {
                    u64 deduplicated_newt_tuples_mem_size =
                        newt->tuple_counts * sizeof(tuple_type);
                    checkCuda(cudaMalloc((void **)&deduplicated_newt_tuples,
                                         deduplicated_newt_tuples_mem_size));
                    checkCuda(cudaMemset(deduplicated_newt_tuples, 0,
                                         deduplicated_newt_tuples_mem_size));
                    //////
                    tuple_type *deuplicated_end = thrust::set_difference(
                        thrust::device, newt->tuples,
                        newt->tuples + newt->tuple_counts, tuple_full,
                        tuple_full + current_full_size,
                        deduplicated_newt_tuples,
                        tuple_indexed_less(full->index_column_size,
                                           full->arity));
                    // checkCuda(cudaDeviceSynchronize());
                    deduplicate_size =
                        deuplicated_end - deduplicated_newt_tuples;
                }

                if (deduplicate_size == 0) {
                    free_relation_container(newt);
                    delta = new GHashRelContainer(rel->arity,
                                                  rel->index_column_size,
                                                  rel->dependent_column_size);
                    continue;
                }

                timer.stop_timer();
                set_diff_time += timer.get_spent_time();

                column_type *deduplicated_raw;
                u64 dedeuplicated_raw_mem_size =
                    deduplicate_size * newt->arity * sizeof(column_type);
                checkCuda(cudaMalloc((void **)&deduplicated_raw,
                                     dedeuplicated_raw_mem_size));
                checkCuda(cudaMemset(deduplicated_raw, 0,
                                     dedeuplicated_raw_mem_size));
                flatten_tuples_raw_data<<<grid_size, block_size>>>(
                    deduplicated_newt_tuples, deduplicated_raw,
                    deduplicate_size, newt->arity);
                checkCuda(cudaGetLastError());
                checkCuda(cudaDeviceSynchronize());
                checkCuda(cudaFree(deduplicated_newt_tuples));

                free_relation_container(newt);

                timer.start_timer();
                float load_detail_time[5] = {0, 0, 0, 0, 0};
                delta =
                    new GHashRelContainer(rel->arity, rel->index_column_size,
                                          rel->dependent_column_size);
                load_relation_container(
                    delta, full->arity, deduplicated_raw, deduplicate_size,
                    full->index_column_size, full->dependent_column_size,
                    full->index_map_load_factor, grid_size, block_size,
                    load_detail_time, true, true, true);
                // checkCuda(cudaDeviceSynchronize());
                timer.stop_timer();
                rebuild_delta_time += timer.get_spent_time();
                rebuild_rel_sort_time += load_detail_time[0];
                rebuild_rel_unique_time += load_detail_time[1];
                rebuild_rel_index_time += load_detail_time[2];

                // auto old_full = rel->tuple_full;
                float flush_detail_time[5] = {0, 0, 0, 0, 0};
                timer.start_timer();
                rel->flush_delta(grid_size, block_size, flush_detail_time);
                timer.stop_timer();
                merge_time += flush_detail_time[1];
                memory_alloc_time += flush_detail_time[0];
                memory_alloc_time += flush_detail_time[2];
                // checkCuda(cudaFree(old_full));
            }

            tuple_size_t all_deduplicate_size = 0;
            tuple_size_t current_rank_delta_size = 0;
            for (auto &delta : rel->deltas) {
                current_rank_delta_size += delta->tuple_counts;
            }
            all_deduplicate_size = current_rank_delta_size;
            if (mcomm->isInitialized()) {
                // multi GPU, reduce deduplicate size
                all_deduplicate_size =
                    mcomm->reduceSumTupleSize(all_deduplicate_size);
            }

            if (all_deduplicate_size != 0) {
                fixpoint_flag = false;
            }

            // print_tuple_rows(rel->full, "Path full after load newt");
            if (verbose_log &&
                (!mcomm->isInitialized() || mcomm->getRank() == 0)) {
                std::cout << "iteration " << iteration_counter << " relation "
                          << rel->name << " rank " << mcomm->getRank()
                          << " delta tuple size: "
                          << current_rank_delta_size
//                             << " full counts " << rel->current_full_size
                          << std::endl;
            }
        }
        if (verbose_log && (!mcomm->isInitialized() || mcomm->getRank() == 0)) {
            std::cout << "Iteration " << iteration_counter
                      << " finish populating" << std::endl;
            // print_memory_usage();
            std::cout << "Join time: " << join_time
                      << " ; merge full time: " << merge_time
                      << " ; memory alloc time: " << memory_alloc_time
                      << " ; rebuild delta time: " << rebuild_delta_time
                      << " ; set diff time: " << set_diff_time
                      << " ; node comm time: "
                      << node_comm_time - prev_comm_time << std::endl;
        }
        iteration_counter++;

        if (fixpoint_flag || iteration_counter > max_iteration) {
            if (!mcomm->isInitialized() || mcomm->getRank() == 0) {
                std::cout << "Iteration " << iteration_counter
                          << " finish populating" << std::endl;
                // print_memory_usage();
                std::cout << "Join time: " << join_time
                          << " ; merge full time: " << merge_time
                          << " ; memory alloc time: " << memory_alloc_time
                          << " ; rebuild delta time: " << rebuild_delta_time
                          << " ; set diff time: " << set_diff_time
                          << " ; node comm time: "
                          << node_comm_time - prev_comm_time << std::endl;
            }
            break;
        }
    }
    // merge full after reach fixpoint
    timer.start_timer();
    if (reload_full_flag) {
        if (!mcomm->isInitialized() || mcomm->getRank() == 0) {
            std::cout << "Start merge full" << std::endl;
        }
        for (Relation *rel : update_relations) {
            // if (rel->current_full_size <= rel->full->tuple_counts) {
            //     continue;
            // }
            auto full = rel->fulls[0];
            auto delta = rel->deltas[0];
            auto newt = rel->newts[0];
            auto tuple_full = rel->tuple_fulls[0];
            auto current_full_size = rel->current_full_sizes[0];
            for (int b_id = 0; b_id < rel->sub_bucket_size; b_id++) {
                column_type *new_full_raw_data;
                u64 new_full_raw_data_mem_size = current_full_size *
                                                 full->arity *
                                                 sizeof(column_type);
                checkCuda(cudaMalloc((void **)&new_full_raw_data,
                                     new_full_raw_data_mem_size));
                checkCuda(cudaMemset(new_full_raw_data, 0,
                                     new_full_raw_data_mem_size));
                flatten_tuples_raw_data<<<grid_size, block_size>>>(
                    tuple_full, new_full_raw_data, current_full_size,
                    full->arity);
                checkCuda(cudaGetLastError());
                checkCuda(cudaDeviceSynchronize());
                // cudaFree(tuple_merge_buffer);
                float load_detail_time[5] = {0, 0, 0, 0, 0};
                load_relation_container(
                    full, full->arity, new_full_raw_data,
                    current_full_size, full->index_column_size,
                    full->dependent_column_size,
                    full->index_map_load_factor, grid_size, block_size,
                    load_detail_time, true, true, true);
                checkCuda(cudaDeviceSynchronize());
                rebuild_rel_sort_time += load_detail_time[0];
                rebuild_rel_unique_time += load_detail_time[1];
                rebuild_rel_index_time += load_detail_time[2];
                // std::cout << "Finished! " << rel->name << " has "
                //           << rel->full->tuple_counts << std::endl;
                for (auto &delta_b : rel->buffered_delta_vectors) {
                    free_relation_container(delta_b);
                }
                free_relation_container(delta);
                free_relation_container(newt);
            }
        }
    } else {
        if (!mcomm->isInitialized() || mcomm->getRank() == 0) {
            for (Relation *rel : update_relations) {
                auto total_full_size = 0;
                for (auto &full : rel->fulls) {
                    total_full_size += full->tuple_counts;
                }
                std::cout << "Finished! " << rel->name << " has "
                          << total_full_size << std::endl;
            }
        }
    }
    timer.stop_timer();
    float merge_full_time = timer.get_spent_time();

    if (!mcomm->isInitialized() || mcomm->getRank() == 0) {
        std::cout << "Join time: " << join_time
                  << " ; merge full time: " << merge_time
                  << " ; rebuild full time: " << merge_full_time
                  << " ; rebuild delta time: " << rebuild_delta_time
                  << " ; set diff time: " << set_diff_time << std::endl;
        std::cout << "Rebuild relation detail time : rebuild rel sort time: "
                  << rebuild_rel_sort_time
                  << " ; rebuild rel unique time: " << rebuild_rel_unique_time
                  << " ; rebuild rel index time: " << rebuild_rel_index_time
                  << " ; node comm time: " << node_comm_time << std::endl;
    }
}
