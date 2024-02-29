// MPI communication

#pragma once

#include "../include/relation.cuh"
#include <mpi.h>
#include <thrust/device_vector.h>

#ifndef USE_64_BIT_TUPLE
#define MPI_ELEM_TYPE MPI_UINT32_T
#else
#define MPI_ELEM_TYPE MPI_UINT64_T
#endif

inline int get_current_rank() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

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
    void distribute_bucket(GHashRelContainer *rel_container);

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

    // rebalance sub buckets inside a relation
    void rebalance(Relation *rel);

    int device_id;
    int grid_size;
    int block_size;

    int rebalance_threshold = 4;
    int split_constant = 2;

  private:
    int rank;
    int total_rank = 0;
    MPI_Comm comm;
    MPI_Status status;
#ifdef DEFAULT_GPU_RDMA
    bool gpu_direct_flag = true;
#else
    bool gpu_direct_flag = false;
#endif
    bool is_initialized = false;

    // persitent buffer avoid allocation overhead
    // send and receive buffer
    // thrust::device_vector<column_type> send_buffer;
    // thrust::device_vector<column_type> recv_buffer;

    thrust::device_vector<uint8_t> tuple_rank_mapping;

    // compute the sub bucket of each tuple inside a container
    // NOTE: this function must be called after sub bucket map in
    // relation has been assign
    void
    computeSubBucket(GHashRelContainer *rel_container,
                     bucket_map_t &sub_bucket_map,
                     thrust::device_vector<bucket_id_t> &tuple_subbucket_map);

    // compute new sub bucket size for a relation
    void computeNewSubBucketSize(Relation *rel);

    // compute the sub bucket map of a relation
    void computeSubBucketMap(Relation *rel);

    // distribute a container based on a bucket to rank mapping
    // and bucket communication is done via bucket size equal to
    // rank size, and mapping is sequecne from 0 ~ rank_size
    void distribute_by_rank_mapping(
        GHashRelContainer *rel_container, bucket_map_t &bucket_map,
        thrust::device_vector<bucket_id_t> &tuple_bucket_map);
};
