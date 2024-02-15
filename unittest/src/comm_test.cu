
// test for comm.h

#include "comm.h"
#include "exception.cuh"
#include "relation.cuh"
#include "relational_algebra.cuh"
#include "print.cuh"
#include <iostream>

column_type raw_graph_data[20] = {1, 2, 1, 5, 1, 6, 2, 3, 2, 6,
                                  3, 4, 8, 7, 4, 5, 4, 6, 5, 6};
tuple_size_t graph_edge_counts = 10;

bool test_split_relation(int argc, char **argv) {
    // test for comm.h
    Communicator comm;
    int device_id;
    int number_of_sm;
    checkCuda(cudaGetDevice(&device_id));
    checkCuda(cudaDeviceGetAttribute(
        &number_of_sm, cudaDevAttrMultiProcessorCount, device_id));
    int block_size = 512;
    int grid_size = 32 * number_of_sm;
    comm.init(argc, argv);
    comm.barrier();
    Relation *path_2__1_2 = new Relation();
    load_relation(path_2__1_2, "path_2__1_2", 2, raw_graph_data,
                  graph_edge_counts, 1, 0, grid_size, block_size);
    // split full relation to all ranks
    comm.distribute(path_2__1_2->full);
    comm.barrier();
    // std::cout << "Rank " << comm.getRank() << " path_2__1_2->full->size() ="
    //           << path_2__1_2->full->tuple_counts << std::endl;

    // let each rank sequentially print the full relation
    for (int i = 0; i < comm.getSize(); i++) {
        if (i == comm.getRank()) {
            std::cout << "Rank " << comm.getRank() << " path_2__1_2->full ="
                      << std::endl;
            print_tuple_rows(path_2__1_2->full, "Full ");
        }
        comm.barrier();
    }

    return true;
}

int main(int argc, char **argv) { return 0; }
