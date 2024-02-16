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
    int getSize() const { return total_rank; }

    void init(int argc, char **argv);
    // barrier the world
    void barrier() { MPI_Barrier(MPI_COMM_WORLD); };

    // distribute relation to all processes by hashing of join column
    void distribute(GHashRelContainer *rel_container);

  int device_id;
  int grid_size;
  int block_size;

  private:
    int rank;
    int total_rank;
    MPI_Comm comm;
    MPI_Status status;

    // persitent buffer avoid allocation overhead
    // send and receive buffer
    thrust::device_vector<column_type> send_buffer;
    thrust::device_vector<column_type> recv_buffer;
    
    thrust::device_vector<uint8_t> tuple_rank_mapping;
};
