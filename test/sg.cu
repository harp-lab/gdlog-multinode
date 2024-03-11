#include <chrono>
#include <fstream>
#include <iostream>
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

//////////////////////////////////////////////////////

long int get_row_size(const char *data_path) {
    std::ifstream f;
    f.open(data_path);
    char c;
    long i = 0;
    while (f.get(c))
        if (c == '\n')
            ++i;
    f.close();
    return i;
}

enum ColumnT { U64, U32 };

column_type *get_relation_from_file(const char *file_path, int total_rows,
                                    int total_columns, char separator,
                                    ColumnT ct) {
    column_type *data =
        (column_type *)malloc(total_rows * total_columns * sizeof(column_type));
    FILE *data_file = fopen(file_path, "r");
    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < total_columns; j++) {
            if (j != (total_columns - 1)) {
                if (ct == U64) {
                    fscanf(data_file, "%lld%c", &data[(i * total_columns) + j],
                           &separator);
                } else {
                    fscanf(data_file, "%ld%c", &data[(i * total_columns) + j],
                           &separator);
                }
            } else {
                if (ct == U64) {
                    fscanf(data_file, "%lld", &data[(i * total_columns) + j]);
                } else {
                    fscanf(data_file, "%ld", &data[(i * total_columns) + j]);
                }
            }
        }
    }
    return data;
}

//////////////////////////////////////////////////////////////////

void analysis_bench(int argc, char *argv[], int block_size, int grid_size) {
    KernelTimer timer;
    Communicator comm;
    comm.init(argc, argv);
    auto dataset_path = argv[1];
    // load the raw graph
    tuple_size_t graph_edge_counts = get_row_size(dataset_path);
    // u64 graph_edge_counts = 2100;
    column_type *raw_graph_data =
        get_relation_from_file(dataset_path, graph_edge_counts, 2, '\t', U32);
    column_type *raw_reverse_graph_data =
        (column_type *)malloc(graph_edge_counts * 2 * sizeof(column_type));
    for (tuple_size_t i = 0; i < graph_edge_counts; i++) {
        raw_reverse_graph_data[i * 2 + 1] = raw_graph_data[i * 2];
        raw_reverse_graph_data[i * 2] = raw_graph_data[i * 2 + 1];
    }

    timer.start_timer();
    Relation *edge_2__1_2 = new Relation();
    Relation *edge_2__2_1 = new Relation();
    load_relation(edge_2__2_1, "edge_2__2_1", 2, raw_reverse_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);
    Relation *sg_2__1_2 = new Relation();
    sg_2__1_2->index_flag = false;
    load_relation(sg_2__1_2, "sg_2__2_1", 2, nullptr, 0, 1, 0, grid_size,
                  block_size);
    load_relation(edge_2__1_2, "edge_2__1_2", 2, raw_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);
    timer.stop_timer();
    float join_detail[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    timer.start_timer();
    LIE init_scc(grid_size, block_size);
    init_scc.set_communicator(&comm);
    init_scc.add_relations(edge_2__1_2, true);
    init_scc.add_relations(sg_2__1_2, false);
    // sg(x, y) :- edge(p, x), edge(p, y), x != y.
    // sg:y,x
    init_scc.add_ra(RelationalJoin(edge_2__1_2, FULL, edge_2__1_2, FULL,
                                   sg_2__1_2, TupleGenerator(2, 2, {1, 3}),
                                   grid_size, block_size, join_detail));
    init_scc.add_ra(RelationalFilter(
        sg_2__1_2, NEWT,
        TupleFilter(2,
                    {BinaryFilterComparison::NE, BinaryFilterComparison::EMPTY},
                    {0, EMPTY_COLUMN}, {1, EMPTY_COLUMN})));

    init_scc.fixpoint_loop();
    timer.stop_timer();

    tuple_size_t total_sg_init_size =
        comm.gatherRelContainerSize(sg_2__1_2->full);
    if (comm.getRank() == 0) {
        std::cout << "sg init counts " << sg_2__1_2->full->tuple_counts
                  << std::endl;
        std::cout << "sg init time: " << timer.get_spent_time() << std::endl;
    }

    LIE sg_lie(grid_size, block_size);
    sg_lie.set_communicator(&comm);
    Relation *tmp = new Relation();
    load_relation(tmp, "tmp", 2, nullptr, 0, 1, 0, grid_size, block_size);
    tmp->index_flag = false;
    sg_lie.add_relations(edge_2__1_2, true);
    sg_lie.add_relations(sg_2__1_2, false);

    sg_lie.add_tmp_relation(tmp);
    // sg(x, y) :- edge(a, x), sg(a, b), edge(b, y).
    // tmp(b,x) :- edge(a, x), sg(a, b).
    sg_lie.add_ra(RelationalJoin(edge_2__1_2, FULL, sg_2__1_2, DELTA, tmp,
                                 TupleGenerator(2, 2, {3, 1}), grid_size,
                                 block_size, join_detail));
    sg_lie.add_ra(RelationalSync(tmp, NEWT));
    // sg(x, y) :- edge(b, y), tmp(b, x).
    sg_lie.add_ra(RelationalJoin(edge_2__1_2, FULL, tmp, NEWT, sg_2__1_2,
                                 TupleGenerator(2, 2, {3, 1}), grid_size,
                                 block_size, join_detail));
    timer.start_timer();
    sg_lie.fixpoint_loop();
    timer.stop_timer();
    total_sg_init_size = comm.gatherRelContainerSize(sg_2__1_2->full);
    if (comm.getRank() == 0) {
        std::cout << "sg counts " << total_sg_init_size << std::endl;
        std::cout << "sg time: " << timer.get_spent_time() << std::endl;
    }
}

int main(int argc, char *argv[]) {
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount,
                           device_id);
    int block_size, grid_size;
    block_size = 512;
    grid_size = 32 * number_of_sm;
    std::locale loc("");

    analysis_bench(argc, argv, block_size, grid_size);
    return 0;
}
