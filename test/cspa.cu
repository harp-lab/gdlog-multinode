#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <vector>

#include "../include/builtin.h"
#include "../include/exception.cuh"
#include "../include/lie.cuh"
#include "../include/print.cuh"
#include "../include/timer.cuh"

void analysis_bench(int argc, char *argv[], int block_size, int grid_size) {
    KernelTimer timer;
    int relation_columns = 2;
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    auto dataset_path = argv[1];
    double spent_time;
    Communicator comm;
    comm.init(argc, argv);

    // load the input relation
    std::stringstream assign_fact_ss;
    assign_fact_ss << dataset_path << "/assign.facts";
    std::stringstream dereference_fact_ss;
    dereference_fact_ss << dataset_path << "/dereference.facts";
    thrust::host_vector<column_type> raw_assign_vec;
    std::map<column_type, std::string> string_map;
    file_to_buffer(dataset_path, raw_assign_vec, string_map);
    tuple_size_t assign_counts = raw_assign_vec.size() / 2;
    column_type *raw_assign_data = raw_assign_vec.data();
    
    column_type *raw_reverse_assign_data =
        (column_type *)malloc(assign_counts * 2 * sizeof(column_type));
    for (tuple_size_t i = 0; i < assign_counts; i++) {
        raw_reverse_assign_data[i * 2 + 1] = raw_assign_data[i * 2];
        raw_reverse_assign_data[i * 2] = raw_assign_data[i * 2 + 1];
    }

    // tuple_size_t dereference_counts =
    //     get_row_size(dereference_fact_ss.str().c_str());
    // column_type *raw_dereference_data = get_relation_from_file(
    //     dereference_fact_ss.str().c_str(), dereference_counts, 2, '\t', U32);
    thrust::host_vector<column_type> raw_dereference_vec;
    file_to_buffer(dataset_path, raw_dereference_vec, string_map);
    tuple_size_t dereference_counts = raw_dereference_vec.size() / 2;
    column_type *raw_dereference_data = raw_dereference_vec.data();
    column_type *raw_reverse_dereference_data =
        (column_type *)malloc(dereference_counts * 2 * sizeof(column_type));
    for (tuple_size_t i = 0; i < dereference_counts; i++) {
        raw_reverse_dereference_data[i * 2 + 1] = raw_dereference_data[i * 2];
        raw_reverse_dereference_data[i * 2] = raw_dereference_data[i * 2 + 1];
    }

    timer.start_timer();

    Relation *assign_2__2_1 = new Relation();
    load_relation(assign_2__2_1, "assign_2__2_1", 2, raw_reverse_assign_data,
                  assign_counts, 1, 0, grid_size, block_size);

    Relation *dereference_2__1_2 = new Relation();
    load_relation(dereference_2__1_2, "dereference_2__1_2", 2,
                  raw_dereference_data, dereference_counts, 1, 0, grid_size,
                  block_size);
    Relation *dereference_2__2_1 = new Relation();
    load_relation(dereference_2__2_1, "dereference_2__2_1", 2,
                  raw_reverse_dereference_data, dereference_counts, 1, 0,
                  grid_size, block_size);
    timer.stop_timer();

    // scc init
    Relation *value_flow_2__1_2 = new Relation();
    load_relation(value_flow_2__1_2, "value_flow_2__1_2", 2, nullptr, 0, 1, 0,
                  grid_size, block_size);
    Relation *value_flow_2__2_1 = new Relation();
    load_relation(value_flow_2__2_1, "value_flow_2__2_1", 2, nullptr, 0, 1, 0,
                  grid_size, block_size);

    Relation *memory_alias_2__1_2 = new Relation();
    load_relation(memory_alias_2__1_2, "memory_alias_2__1_2", 2, nullptr, 0, 1,
                  0, grid_size, block_size);
    Relation *memory_alias_2__2_1 = new Relation();
    load_relation(memory_alias_2__2_1, "memory_alias_2__2_1", 2, nullptr, 0, 1,
                  0, grid_size, block_size);

    timer.start_timer();
    time_point_begin = std::chrono::high_resolution_clock::now();
    LIE init_scc(grid_size, block_size);
    init_scc.set_communicator(&comm);
    init_scc.add_relations(value_flow_2__1_2, false);
    init_scc.add_relations(value_flow_2__2_1, false);
    init_scc.add_relations(memory_alias_2__1_2, false);
    init_scc.add_relations(memory_alias_2__2_1, false);
    init_scc.add_relations(assign_2__2_1, true);

    init_scc.add_ra(RelationalCopy(assign_2__2_1, FULL, value_flow_2__1_2,
                                   TupleProjector(2, {0, 0}), grid_size,
                                   block_size));
    init_scc.add_ra(RelationalCopy(assign_2__2_1, FULL, value_flow_2__1_2,
                                   TupleProjector(2, {1, 1}), grid_size,
                                   block_size));
    init_scc.add_ra(RelationalCopy(assign_2__2_1, FULL, value_flow_2__1_2,
                                   TupleProjector(2, {1, 0}), grid_size,
                                   block_size));

    init_scc.add_ra(RelationalCopy(assign_2__2_1, FULL, memory_alias_2__1_2,
                                   TupleProjector(2, {0, 0}), grid_size,
                                   block_size));
    init_scc.add_ra(RelationalCopy(assign_2__2_1, FULL, memory_alias_2__1_2,
                                   TupleProjector(2, {1, 1}), grid_size,
                                   block_size));

    init_scc.add_ra(RelationalCopy(value_flow_2__1_2, DELTA, value_flow_2__2_1,
                                   TupleProjector(2, {1, 0}), grid_size,
                                   block_size));
    init_scc.add_ra(
        RelationalCopy(memory_alias_2__1_2, DELTA, memory_alias_2__2_1,
                       TupleProjector(2, {1, 0}), grid_size, block_size));
    init_scc.fixpoint_loop();

    timer.stop_timer();
    time_point_end = std::chrono::high_resolution_clock::now();
    if (comm.getRank() == 0) {
        std::cout << "init scc time: " << timer.get_spent_time() << std::endl;
        std::cout << "init scc time (chono): "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         time_point_end - time_point_begin)
                         .count()
                  << std::endl;
    }

    // scc analysis
    Relation *value_flow_forward_2__1_2 = new Relation();
    load_relation(value_flow_forward_2__1_2, "value_flow_forward_2__1_2", 2,
                  nullptr, 0, 1, 0, grid_size, block_size);

    Relation *value_flow_forward_2__2_1 = new Relation();
    load_relation(value_flow_forward_2__2_1, "value_flow_forward_2__2_1", 2,
                  nullptr, 0, 1, 0, grid_size, block_size);

    Relation *value_alias_2__1_2 = new Relation();
    value_alias_2__1_2->index_flag = false;
    load_relation(value_alias_2__1_2, "value_alias_2__1_2", 2, nullptr, 0, 1, 0,
                  grid_size, block_size);

    Relation *tmp_rel_def = new Relation();
    tmp_rel_def->index_flag = false;
    load_relation(tmp_rel_def, "tmp_rel_def", 2, nullptr, 0, 1, 0, grid_size,
                  block_size);
    Relation *tmp_rel_ma1 = new Relation();
    tmp_rel_ma1->index_flag = false;
    load_relation(tmp_rel_ma1, "tmp_rel_ma1", 2, nullptr, 0, 1, 0, grid_size,
                  block_size, true);
    Relation *tmp_rel_ma2 = new Relation();
    tmp_rel_ma2->index_flag = false;
    load_relation(tmp_rel_ma2, "tmp_rel_ma2", 2, nullptr, 0, 1, 0, grid_size,
                  block_size, true);

    LIE analysis_scc(grid_size, block_size);
    analysis_scc.set_communicator(&comm);
    analysis_scc.add_relations(assign_2__2_1, true);
    analysis_scc.add_relations(dereference_2__1_2, true);
    analysis_scc.add_relations(dereference_2__2_1, true);

    analysis_scc.add_relations(value_flow_2__1_2, false);
    analysis_scc.add_relations(value_flow_2__2_1, false);
    analysis_scc.add_relations(memory_alias_2__1_2, false);
    analysis_scc.add_relations(memory_alias_2__2_1, false);
    analysis_scc.add_relations(value_alias_2__1_2, false);

    // join order matters for temp!
    analysis_scc.add_tmp_relation(tmp_rel_def);
    analysis_scc.add_tmp_relation(tmp_rel_ma1);
    analysis_scc.add_tmp_relation(tmp_rel_ma2);

    float join_detail[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    // join_vf_vfvf: ValueFlow(x, y) :- ValueFlow(x, z), ValueFlow(z, y).
    analysis_scc.add_ra(RelationalJoin(
        value_flow_2__1_2, FULL, value_flow_2__2_1, DELTA, value_flow_2__1_2,
        TupleGenerator(2, 2, {3, 1}), grid_size, block_size, join_detail));
    analysis_scc.add_ra(RelationalJoin(
        value_flow_2__2_1, FULL, value_flow_2__1_2, DELTA, value_flow_2__1_2,
        TupleGenerator(2, 2, {1, 3}), grid_size, block_size, join_detail));

    // join_va_vf_vf: ValueAlias(x, y) :- ValueFlow(z, x), ValueFlow(z, y).
    // v1
    analysis_scc.add_ra(RelationalJoin(
        value_flow_2__1_2, FULL, value_flow_2__1_2, DELTA, value_alias_2__1_2,
        TupleGenerator(2, 2, {1, 3}), grid_size, block_size, join_detail));
    // v2
    analysis_scc.add_ra(RelationalJoin(
        value_flow_2__1_2, FULL, value_flow_2__1_2, DELTA, value_alias_2__1_2,
        TupleGenerator(2, 2, {3, 1}), grid_size, block_size, join_detail));

    // join_vf_am: ValueFlow(x, y) :- Assign(x, z), MemoryAlias(z, y).
    analysis_scc.add_ra(RelationalJoin(
        assign_2__2_1, FULL, memory_alias_2__1_2, DELTA, value_flow_2__1_2,
        TupleGenerator(2, 2, {1, 3}), grid_size, block_size, join_detail));

    // tmp_rel_def(z, x) :- Dereference(y, x), ValueAlias(y, z)
    analysis_scc.add_ra(RelationalJoin(
        dereference_2__1_2, FULL, value_alias_2__1_2, DELTA, tmp_rel_def,
        TupleGenerator(2, 2, {3, 1}), grid_size, block_size, join_detail));
    analysis_scc.add_ra(RelationalSync(tmp_rel_def, NEWT));

    // WARNING: tmp relation can only in outer because it doesn't include
    // index!
    // join_ma_d_tmp: MemoryAlias(x, w) :- Dereference(z, w) , tmp_rel_def(z,x)
    analysis_scc.add_ra(RelationalJoin(
        dereference_2__1_2, FULL, tmp_rel_def, NEWT, memory_alias_2__1_2,
        TupleGenerator(2, 2, {3, 1}), grid_size, block_size, join_detail));

    // ValueAlias(x,y) :-
    //    ValueFlow(z,x),
    //    MemoryAlias(z,w),
    //    ValueFlow(w,y).
    // ValueFlow DELTA 1, 2 <> MemoryAlias FULL 1, 2 <> ValueFlow FULL 2, 1
    // ValueFlow FULL 1, 2 <> MemoryAlias DELTA 1, 2 <> ValueFlow FULL 2, 1
    // ValueFlow FULL 1, 2 <> MemoryAlias FULL 1, 2 <> ValueFlow DELTA 2, 1
    // join_tmp_vf_ma : tmp_rel_ma(w, x) :- ValueFlow(z, x), MemoryAlias(z, w).
    // join_va_tmp_vf : ValueAlias(x, y) :- tmp_rel_ma(w, x), ValueFlow(w,y).
    // v1
    analysis_scc.add_ra(RelationalJoin(
        memory_alias_2__1_2, FULL, value_flow_2__1_2, DELTA, tmp_rel_ma1,
        TupleGenerator(2, 2, {1, 3}), grid_size, block_size, join_detail));
    analysis_scc.add_ra(RelationalJoin(
        value_flow_2__1_2, FULL, memory_alias_2__1_2, DELTA, tmp_rel_ma1,
        TupleGenerator(2, 2, {3, 1}), grid_size, block_size, join_detail));
    analysis_scc.add_ra(RelationalSync(tmp_rel_ma1, NEWT));

    analysis_scc.add_ra(RelationalJoin(
        value_flow_2__1_2, FULL, tmp_rel_ma1, NEWT, value_alias_2__1_2,
        TupleGenerator(2, 2, {3, 1}), grid_size, block_size, join_detail));

    analysis_scc.add_ra(RelationalJoin(
        memory_alias_2__2_1, FULL, value_flow_2__1_2, DELTA, tmp_rel_ma2,
        TupleGenerator(2, 2, {1, 3}), grid_size, block_size, join_detail));
    analysis_scc.add_ra(RelationalSync(tmp_rel_ma2, NEWT));
    analysis_scc.add_ra(RelationalJoin(
        value_flow_2__1_2, FULL, tmp_rel_ma2, NEWT, value_alias_2__1_2,
        TupleGenerator(2, 2, {1, 3}), grid_size, block_size, join_detail));

    analysis_scc.add_ra(RelationalACopy(value_flow_2__1_2, value_flow_2__2_1,
                                        TupleProjector(2, {1, 0}), grid_size,
                                        block_size));
    analysis_scc.add_ra(
        RelationalACopy(memory_alias_2__1_2, memory_alias_2__2_1,
                        TupleProjector(2, {1, 0}), grid_size, block_size));
    time_point_begin = std::chrono::high_resolution_clock::now();
    timer.start_timer();
    analysis_scc.fixpoint_loop();
    // print_tuple_rows(value_flow_2__1_2->full, "value_flow_2__1_2");
    timer.stop_timer();
    auto total_value_flow = value_flow_2__1_2->full->tuple_counts;
    total_value_flow = comm.reduceSumTupleSize(total_value_flow);
    if (comm.getRank() == 0) {
        std::cout << "total value flow: " << total_value_flow << std::endl;
        time_point_end = std::chrono::high_resolution_clock::now();
        std::cout << "analysis scc time: " << timer.get_spent_time()
                  << std::endl;
        std::cout << "analysis scc time (chono): "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         time_point_end - time_point_begin)
                         .count()
                  << std::endl;
        std::cout << "join detail: " << std::endl;
        std::cout << "compute size time:  " << join_detail[0] << std::endl;
        std::cout << "reduce + scan time: " << join_detail[1] << std::endl;
        std::cout << "fetch result time:  " << join_detail[2] << std::endl;
        std::cout << "sort time:          " << join_detail[3] << std::endl;
        std::cout << "build index time:   " << join_detail[5] << std::endl;
        std::cout << "merge time:         " << join_detail[6] << std::endl;
        std::cout << "unique time:        " << join_detail[4] + join_detail[7]
                  << std::endl;
    }
}

int main(int argc, char *argv[]) {
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount,
                           device_id);
    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block,
                           cudaDevAttrMaxThreadsPerBlock, 0);
    int block_size, grid_size;
    block_size = 512;
    grid_size = 32 * number_of_sm;
    std::locale loc("");
    analysis_bench(argc, argv, block_size, grid_size);
    return 0;
}
