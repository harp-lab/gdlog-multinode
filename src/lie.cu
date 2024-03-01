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

void LIE::add_ra(ra_op op, bool is_iterative) {
    if (is_iterative) {
        ra_ops.push_back(op);
    } else {
        non_iterative_ra_ops.push_back(op);
    }
}

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
            mcomm->distribute(rel->full);
            rel->full->build_index(grid_size, block_size);
        }
        for (Relation *rel : static_relations) {
            mcomm->distribute(rel->full);
            rel->full->build_index(grid_size, block_size);
        }
    }
}

void LIE::execute_ra(ra_op &ra) {
    std::visit(
        dynamic_dispatch{
            [&](RelationalJoin &op) {
                // timer.start_timer();
                op();
            },
            [&](RelationalACopy &op) { op(); },
            [&](RelationalCopy &op) {
                // std::cout << "copied" << std::endl;
                if (op.src_ver == FULL) {
                    if (!op.copied) {
                        op();
                        op.copied = true;
                    }
                } else {
                    op();
                }
            },
            [&](RelationalFilter &op) { op(); },
            [&](RelationalArithm &op) { op(); },
            [&](RelationalNegation &op) { op(); },
            [&](RelationalSync &op) {
                if (op.src_ver == FULL) {
                    mcomm->distribute(op.src_rel->full);
                } else if (op.src_ver == DELTA) {
                    mcomm->distribute(op.src_rel->delta);
                } else {
                    // std::cout << ">>>>>>>>>>>>>>>>>> sync " <<
                    // mcomm->getTotalRank() << std::endl;
                    mcomm->distribute(op.src_rel->newt);
                }
            },
            [&](RelationalIndex &op) {
                if (op.target_ver == FULL) {
                    op.target_rel->full->build_index(grid_size, block_size);
                } else if (op.target_ver == DELTA) {
                    op.target_rel->delta->build_index(grid_size, block_size);
                } else {
                    op.target_rel->newt->build_index(grid_size, block_size);
                }
            },
            [&](RelationalCartesian &op) {
                mcomm->broadcast(get_relation_ver(op.outer_rel, op.outer_ver));
                op.inner_rel->defragement(op.inner_ver, grid_size, block_size);
                // print_tuple_rows(op.inner_rel->full, "inner rel");
                // std::cout << "cartesian" << std::endl;
                op();
            },
            [&](RelationalUnion &op) { op(); },
            [&](RelationalClear &op) { op(); },
            [&](RelationalBroadcast &op) { mcomm->broadcast(op.src); }},
        ra);
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
        rel->delta->free();
        checkCuda(cudaMalloc((void **)&rel->delta->data_raw,
                             rel->full->arity * rel->full->tuple_counts *
                                 sizeof(column_type)));
        checkCuda(cudaMemcpy(rel->delta->data_raw, rel->full->data_raw,
                             rel->full->arity * rel->full->tuple_counts *
                                 sizeof(column_type),
                             cudaMemcpyDeviceToDevice));
        float detail_time[5];
        load_relation_container(
            rel->delta, rel->full->arity, rel->delta->data_raw,
            rel->full->tuple_counts, rel->full->index_column_size,
            rel->full->dependent_column_size, 0.8, grid_size, block_size,
            detail_time, true, true, true);
        checkCuda(cudaDeviceSynchronize());
    }
    while (true) {
        auto prev_comm_time = node_comm_time;
        if (iteration_counter == 0) {
            for (auto &ra_op : non_iterative_ra_ops) {
                timer.start_timer();
                execute_ra(ra_op);
                timer.stop_timer();
                join_time += timer.get_spent_time();
            }
        }
        for (auto &ra_op : ra_ops) {
            timer.start_timer();
            execute_ra(ra_op);
            timer.stop_timer();
            join_time += timer.get_spent_time();
        }

        // clean tmp relation
        for (Relation *rel : tmp_relations) {
            rel->newt->free();
        }

        // merge delta into full
        bool fixpoint_flag = true;
        for (Relation *rel : update_relations) {

            if (iteration_counter == 0) {
                rel->delta->free();
            }

            // drop the index of delta once merged, because it won't be used in
            // next iter when migrate more general case, this operation need to
            // be put off to end of all RA operation in current iteration
            if (rel->delta->index_map != nullptr) {
                checkCuda(cudaFree(rel->delta->index_map));
                rel->delta->index_map = nullptr;
            }
            if (rel->delta->tuples != nullptr) {
                checkCuda(cudaFree(rel->delta->tuples));
                rel->delta->tuples = nullptr;
            }

            if (mcomm->isInitialized()) {
                // mutil GPU, distributed newt
                timer.start_timer();
                mcomm->distribute(rel->newt);
                timer.stop_timer();
                node_comm_time += timer.get_spent_time();
                // gather newt size
            }

            tuple_size_t newt_size = rel->newt->tuple_counts;
            timer.start_timer();
            tuple_type *deduplicated_newt_tuples;
            tuple_size_t deduplicate_size = 0;
            if (newt_size != 0) {
                u64 deduplicated_newt_tuples_mem_size =
                    rel->newt->tuple_counts * sizeof(tuple_type);
                checkCuda(cudaMalloc((void **)&deduplicated_newt_tuples,
                                     deduplicated_newt_tuples_mem_size));
                checkCuda(cudaMemset(deduplicated_newt_tuples, 0,
                                     deduplicated_newt_tuples_mem_size));
                //////
                tuple_type *deuplicated_end = thrust::set_difference(
                    thrust::device, rel->newt->tuples,
                    rel->newt->tuples + rel->newt->tuple_counts,
                    rel->full->tuples, rel->full->tuples + rel->full->tuple_counts,
                    deduplicated_newt_tuples,
                    tuple_indexed_less(rel->full->index_column_size,
                                       rel->full->arity -
                                           rel->dependent_column_size));
                // checkCuda(cudaDeviceSynchronize());
                deduplicate_size = deuplicated_end - deduplicated_newt_tuples;
            }

            tuple_size_t all_deduplicate_size = deduplicate_size;
            if (mcomm->isInitialized()) {
                // multi GPU, reduce deduplicate size
                all_deduplicate_size =
                    mcomm->reduceSumTupleSize(deduplicate_size);
            }

            if (all_deduplicate_size != 0) {
                fixpoint_flag = false;
            }

            if (deduplicate_size == 0) {
                rel->newt->free();
                rel->delta =
                    new GHashRelContainer(rel->arity, rel->index_column_size,
                                          rel->dependent_column_size);
                continue;
            }

            timer.stop_timer();
            set_diff_time += timer.get_spent_time();

            column_type *deduplicated_raw;
            u64 dedeuplicated_raw_mem_size =
                deduplicate_size * rel->newt->arity * sizeof(column_type);
            checkCuda(cudaMalloc((void **)&deduplicated_raw,
                                 dedeuplicated_raw_mem_size));
            // checkCuda(
            //     cudaMemset(deduplicated_raw, 0, dedeuplicated_raw_mem_size));
            thrust::for_each(
                thrust::device, thrust::make_counting_iterator<tuple_size_t>(0),
                thrust::make_counting_iterator<tuple_size_t>(deduplicate_size),
                [gh_tps = deduplicated_newt_tuples, arity = rel->newt->arity,
                 new_data = deduplicated_raw] __device__(tuple_size_t i) {
                    for (int j = 0; j < arity; j++) {
                        new_data[i * arity + j] = gh_tps[i][j];
                    }
                });

            checkCuda(cudaFree(deduplicated_newt_tuples));

            rel->newt->free();

            timer.start_timer();
            float load_detail_time[5] = {0, 0, 0, 0, 0};
            rel->delta = new GHashRelContainer(
                rel->arity, rel->index_column_size, rel->dependent_column_size);
            load_relation_container(
                rel->delta, rel->full->arity, deduplicated_raw,
                deduplicate_size, rel->full->index_column_size,
                rel->full->dependent_column_size,
                rel->full->index_map_load_factor, grid_size, block_size,
                load_detail_time, true, true, true);
            // checkCuda(cudaDeviceSynchronize());
            timer.stop_timer();
            rebuild_delta_time += timer.get_spent_time();
            rebuild_rel_sort_time += load_detail_time[0];
            rebuild_rel_unique_time += load_detail_time[1];
            rebuild_rel_index_time += load_detail_time[2];

            float flush_detail_time[5] = {0, 0, 0, 0, 0};
            timer.start_timer();
            rel->flush_delta(grid_size, block_size, flush_detail_time);
            timer.stop_timer();
            merge_time += flush_detail_time[1];
            memory_alloc_time += flush_detail_time[0];
            memory_alloc_time += flush_detail_time[2];
            // checkCuda(cudaFree(old_full));

            // print_tuple_rows(rel->full, "Path full after load newt");
            if (verbose_log &&
                (!mcomm->isInitialized() || mcomm->getRank() == 0)) {
                std::cout << "iteration " << iteration_counter << " relation "
                          << rel->name << " rank " << mcomm->getRank()
                          << " finish dedup new tuples : " << deduplicate_size
                          << " delta tuple size: " << rel->delta->tuple_counts
                          << " full counts " << rel->full->tuple_counts
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
            float load_detail_time[5] = {0, 0, 0, 0, 0};
            rel->defragement(FULL, grid_size, block_size);
            rebuild_rel_sort_time += load_detail_time[0];
            rebuild_rel_unique_time += load_detail_time[1];
            rebuild_rel_index_time += load_detail_time[2];
            // std::cout << "Finished! " << rel->name << " has "
            //           << rel->full->tuple_counts << std::endl;
            // for (auto &delta_b : rel->buffered_delta_vectors) {
            //     delta_b->free();
            // }
            rel->buffered_delta_vectors.clear();
            rel->delta->free();
            rel->newt->free();
        }
    } else {
        if (!mcomm->isInitialized() || mcomm->getRank() == 0) {
            for (Relation *rel : update_relations) {
                std::cout << "Finished! " << rel->name << " has "
                          << rel->full->tuple_counts << std::endl;
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
