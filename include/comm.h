// MPI communication

#pragma once

#include <mpi.h>
#include <thrust/device_vector.h>
#include "../include/relation.cuh"

class Communicator {

  public:
    Communicator(){};
    ~Communicator();

    int getRank() const { return rank; }
    int getSize() const { return size; }

    void init(int argc, char **argv);
    void barrier() { MPI_Barrier(comm); };

    // distribute relation to all processes by hashing of join column
    void distribute(GHashRelContainer *rel_container);

  int device_id;
  int grid_size;
  int block_size;

  private:
    int rank;
    int size;
    MPI_Comm comm;
    MPI_Status status;

    // persitent buffer avoid allocation overhead
    // send and receive buffer
    thrust::device_vector<column_type> send_buffer;
    thrust::device_vector<column_type> recv_buffer;
    
    thrust::device_vector<uint8_t> tuple_rank_mapping;
};
