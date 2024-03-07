
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

void overlap(int argc, char *argv[], int block_size, int grid_size) {
    auto dataset_path = argv[1];
    std::stringstream candidate_bound_path_ss;
    candidate_bound_path_ss << dataset_path
                            << "/block_candidate_boundaries.csv";

    thrust::host_vector<column_type> candidate_boundaries_data;
    std::map<column_type, std::string> string_map;
    file_to_buffer(candidate_bound_path_ss.str(), candidate_boundaries_data,
                   string_map);
    auto candidate_boundaries_data_rows = candidate_boundaries_data.size() / 4;
    std::cout << " >>>>>>>>>>>>> candidate_boud size: "
              << candidate_boundaries_data_rows << std::endl;
    Relation *block_overlap = new Relation();
    load_relation(block_overlap, "block_overlap", 6, nullptr, 0, 1, 0,
                  grid_size, block_size);
    Relation *block_overlap_pred1 = new Relation();
    load_relation(block_overlap_pred1, "block_overlap_pred1", 8, nullptr, 0, 1,
                  0, grid_size, block_size);
    Relation *candidate_boud = new Relation();
    load_relation(candidate_boud, "candidate_boud", 4,
                  candidate_boundaries_data.data(),
                  candidate_boundaries_data_rows, 1, 0, grid_size, block_size);
    Relation *candidate_bound_tmp1 = new Relation();
    load_relation(candidate_bound_tmp1, "candidate_boud_tmp", 5,
                  candidate_boundaries_data.data(), 0, 1, 0, grid_size,
                  block_size);

    // print_tuple_rows(candidate_boud->full, "candidate_boud");

    Communicator comm;
    comm.init(argc, argv);

    LIE overlap_lie(grid_size, block_size);
    overlap_lie.set_communicator(&comm);
    overlap_lie.add_relations(block_overlap, false);
    overlap_lie.add_relations(candidate_boud, true);
    overlap_lie.add_tmp_relation(block_overlap_pred1);
    overlap_lie.add_tmp_relation(candidate_bound_tmp1);

    // candidate_boud(Block1,Type1,BegAddr1,EndAddr1)
    // -->
    // candidate_boud_tmp(Block1,Type1,BegAddr1,EndAddr1, EndAddr1)
    overlap_lie.add_ra(RelationalCopy(candidate_boud, FULL,
                                      candidate_bound_tmp1,
                                      TupleProjector(5, {0, 1, 2, 3, 3}),
                                      grid_size, block_size),
                       false);
    // candidate_boud_tmp(Block1,Type1,BegAddr1,EndAddr1, EndAddr1)
    // -->
    // candidate_boud_tmp(Block1,Type1,BegAddr1,EndAddr1, EndAddr1-BegAddr1)
    overlap_lie.add_ra(
        RelationalArithm(candidate_bound_tmp1, NEWT,
                         TupleArithmetic(5,
                                         {BinaryArithmeticOperator::EMPTY,
                                          BinaryArithmeticOperator::EMPTY,
                                          BinaryArithmeticOperator::EMPTY,
                                          BinaryArithmeticOperator::EMPTY,
                                          BinaryArithmeticOperator::SUB},
                                         {0, 1, 2, 3, 4}, {0, 1, 2, 3, 2})),
        false);
    overlap_lie.add_ra(
        RelationalIndex(candidate_bound_tmp1, NEWT));

    //    block_overlap(Block1,Type1,Size1,Block2,Type2,Size2)
    //     :-
    //    candidate_boud_tmp(Block1,Type1,BegAddr1,EndAddr1,Size1),
    //    candidate_boud_tmp(Block2,Type2,BegAddr2,EndAddr2,Size2),
    //    Block1 != Block2,
    //    BegAddr1 <= BegAddr2,
    //    BegAddr2 < EndAddr1.
    overlap_lie.add_ra(
        RelationalCartesian(candidate_bound_tmp1, NEWT, candidate_bound_tmp1,
                            NEWT, block_overlap,
                            TupleGenerator(6, 5, {0, 1, 4, 5, 6, 9}),
                            TupleJoinFilter(3, 5,
                                            {BinaryFilterComparison::NE,
                                             BinaryFilterComparison::LE,
                                             BinaryFilterComparison::LT},
                                            {0, 2, 7}, {5, 7, 3}),
                            grid_size, block_size),
        false);

    //
    // block_overlap(Block1,Type1,Size1,Block2,Type2,Size2)
    //     :-
    //    candidate_boud_tmp(Block1,Type1,BegAddr1,EndAddr1,Size1),
    //    candidate_boud_tmp(Block2,Type2,BegAddr2,EndAddr2,Size2),
    //    Type1 != Type2,
    //    BegAddr1 <= BegAddr2,
    //    BegAddr2 < EndAddr1.
    overlap_lie.add_ra(
        RelationalCartesian(candidate_bound_tmp1, NEWT, candidate_bound_tmp1,
                            NEWT, block_overlap,
                            TupleGenerator(6, 5, {0, 1, 4, 5, 6, 9}),
                            TupleJoinFilter(3, 5,
                                            {BinaryFilterComparison::NE,
                                             BinaryFilterComparison::LE,
                                             BinaryFilterComparison::LT},
                                            {1, 2, 7}, {6, 7, 3}),
                            grid_size, block_size),
        false);

    // block_overlap(Block1,Type1,AddSub1,Block2,Type2,AddSub2)
    // :-
    // candidate_boud_tmp(Block1,Type1,BegAddr1,EndAddr1, AddSub1)
    // candidate_boud_tmp(Block2,Type2, BegAddr2, EndAddr2, AddSub2)
    // AddSub1 != AddSub2,
    // BegAddr1 <= BegAddr2,
    // BegAddr2 < EndAddr1.
    overlap_lie.add_ra(
        RelationalCartesian(candidate_bound_tmp1, NEWT, candidate_bound_tmp1,
                            NEWT, block_overlap,
                            TupleGenerator(6, 5, {0, 1, 4, 5, 6, 9}),
                            TupleJoinFilter(3, 5,
                                            {BinaryFilterComparison::NE,
                                             BinaryFilterComparison::LE,
                                             BinaryFilterComparison::LT},
                                            {4, 2, 7}, {9, 7, 3}),
                            grid_size, block_size),
        false);

    // block_overlap(Block2,Type2,Size2,Block1,Type1,Size1) :-
    // block_overlap(Block1,Type1,Size1,Block2,Type2,Size2).
    overlap_lie.add_ra(RelationalCopy(block_overlap, NEWT, block_overlap,
                                      TupleProjector(6, {3, 4, 5, 0, 1, 2}),
                                      grid_size, block_size),
                       false);

    KernelTimer timer;
    timer.start_timer();
    overlap_lie.fixpoint_loop();
    timer.stop_timer();
    auto total_block_overlap = block_overlap->full->tuple_counts;
    total_block_overlap = comm.reduceSumTupleSize(total_block_overlap);
    if (comm.getRank() == 0) {
        std::cout << "overlap time: " << timer.get_spent_time() << std::endl;
        std::cout << "total block overlap: " << total_block_overlap
                  << std::endl;
    }
}

int main(int argc, char *argv[]) {
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount,
                           device_id);
    // std::cout << "num of sm " << number_of_sm << std::endl;
    // std::cout << "using " << EMPTY_HASH_ENTRY << " as empty hash entry"
    //           << std::endl;
    int block_size, grid_size;
    block_size = 512;
    grid_size = 32 * number_of_sm;

    overlap(argc, argv, block_size, grid_size);
    return 0;
}
