// MPI communication

#pragma once

#include <mpi.h>
#include <thrust/device_vector.h>
#include "../include/relation.cuh"

#ifndef USE_64_BIT_TUPLE
#define MPI_ELEM_TYPE MPI_UINT32_T
#else
#define MPI_ELEM_TYPE MPI_UINT64_T
#endif

class Communicator {

  public:
    Communicator(){};
    ~Communicator();

    int getRank() const { return rank; }
    int getTotalRank() const { return total_rank; }

    void init(int argc, char **argv);
    // barrier the world
    void barrier() { MPI_Barrier(MPI_COMM_WORLD); };

    // distribute relation to all processes by hashing of join column
    void distribute(GHashRelContainer *rel_container);

    // gather relation size from all processes
    tuple_size_t gatherRelContainerSize(GHashRelContainer *rel_container);
    
    // reduce a bool from all processes
    bool reduceBool(bool value);

    // reduce a tuple_size_t from all processes
    tuple_size_t reduceSumTupleSize(tuple_size_t value);

    void enableGpuDirect() { gpu_direct_flag = true; }
    void disableGpuDirect() { gpu_direct_flag = false; }

    // predicate for gpu direct
    bool isGpuDirect() { return gpu_direct_flag; }

    // predicate for initialized
    bool isInitialized() { return is_initialized; }

  int device_id;
  int grid_size;
  int block_size;

  private:
    int rank;
    int total_rank = 0;
    MPI_Comm comm;
    MPI_Status status;
    bool gpu_direct_flag = true;
    bool is_initialized = false;

    // persitent buffer avoid allocation overhead
    // send and receive buffer
    thrust::device_vector<column_type> send_buffer;
    thrust::device_vector<column_type> recv_buffer;
    
    thrust::device_vector<uint8_t> tuple_rank_mapping;
};
