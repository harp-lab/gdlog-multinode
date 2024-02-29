
#include "../include/comm.h"
#include "../include/exception.cuh"
#include "../include/print.cuh"
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

void Communicator::init(int argc, char **argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &total_rank);
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    is_initialized = true;
}

void Communicator::distribute_bucket(GHashRelContainer *container) {

    // TODO: refactor remove dup code
    // Distribute the data
    // compute the rank of tuple in the container
    if (total_rank == 1) {
        return;
    }
    auto arity = container->arity;

    tuple_rank_mapping.clear();
    tuple_rank_mapping.resize(container->tuple_counts);

    thrust::transform(
        thrust::device, container->tuples,
        container->tuples + container->tuple_counts, tuple_rank_mapping.begin(),
        [total_rank = total_rank, jc = container->index_column_size] __device__(
            const tuple_type &tuple) -> uint8_t {
            return (uint8_t)(prefix_hash(tuple, jc) % total_rank);
        });

    // stable sort the tuples based on the rank
    thrust::stable_sort_by_key(thrust::device, tuple_rank_mapping.begin(),
                               tuple_rank_mapping.end(), container->tuples);
    // tuple size need to send to each rank
    thrust::device_vector<int> rank_tuple_counts(total_rank);
    thrust::device_vector<uint8_t> reduced_rank(total_rank);

    auto reduced_end = thrust::reduce_by_key(
        thrust::device, tuple_rank_mapping.begin(), tuple_rank_mapping.end(),
        thrust::constant_iterator<int>(1), reduced_rank.begin(),
        rank_tuple_counts.begin());
    auto rank_tuple_counts_size = reduced_end.first - reduced_rank.begin();
    rank_tuple_counts.resize(rank_tuple_counts_size);
    reduced_rank.resize(rank_tuple_counts_size);
    // create a host copy of the rank tuple counts and reduced rank
    thrust::host_vector<int> h_rank_tuple_counts(rank_tuple_counts);
    thrust::host_vector<uint8_t> h_reduced_rank(reduced_rank);

    thrust::host_vector<int> h_rank_tuple_send_counts(total_rank);
    for (int i = 0; i < rank_tuple_counts_size; i++) {
        h_rank_tuple_send_counts[h_reduced_rank[i]] = h_rank_tuple_counts[i];
    }
    thrust::host_vector<int> h_rank_tuple_recv_counts(total_rank);

    int total_send = thrust::reduce(h_rank_tuple_send_counts.begin(),
                                    h_rank_tuple_send_counts.end());

    // send the tuple size to each rank
    MPI_Alltoall(h_rank_tuple_send_counts.data(), 1, MPI_INT,
                 h_rank_tuple_recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total_recv = thrust::reduce(h_rank_tuple_recv_counts.begin(),
                                    h_rank_tuple_recv_counts.end());

    // allocate memory for the send and receive buffers
    thrust::device_vector<column_type> d_send_buffer(total_send * arity);

    // copy tuples to the send buffer
    thrust::for_each(
        thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(
            container->tuples, thrust::counting_iterator<uint32_t>(0))),
        thrust::make_zip_iterator(thrust::make_tuple(
            container->tuples + container->tuple_counts,
            thrust::counting_iterator<uint32_t>(container->tuple_counts))),
        [dest = d_send_buffer.data().get(), arity] __device__(
            const thrust::tuple<tuple_type, uint32_t> &t) -> void {
            auto &tuple = thrust::get<0>(t);
            auto &index = thrust::get<1>(t);
            auto dest_tp = dest + index * arity;
            for (int i = 0; i < arity; i++) {
                dest_tp[i] = tuple[i];
            }
        });

    // print send buffer on each rank after copy

    // after send, free the memory  of the container
    free_relation_container(container);
    container->tuple_counts = total_recv;

    // convert the tuple size to column size by times arity on each element
    // in h_rank_tuple_(send/recv)_counts
    thrust::transform(
        h_rank_tuple_send_counts.begin(), h_rank_tuple_send_counts.end(),
        thrust::make_constant_iterator(arity), h_rank_tuple_send_counts.begin(),
        thrust::multiplies<uint32_t>());
    thrust::transform(
        h_rank_tuple_recv_counts.begin(), h_rank_tuple_recv_counts.end(),
        thrust::make_constant_iterator(arity), h_rank_tuple_recv_counts.begin(),
        thrust::multiplies<uint32_t>());

    // create displacements for the send and receive buffers
    thrust::host_vector<int> send_displacements(total_rank);
    thrust::host_vector<int> recv_displacements(total_rank);
    send_displacements[0] = 0;
    recv_displacements[0] = 0;
    for (int i = 1; i < total_rank; i++) {
        send_displacements[i] =
            send_displacements[i - 1] + h_rank_tuple_send_counts[i - 1];
        recv_displacements[i] =
            recv_displacements[i - 1] + h_rank_tuple_recv_counts[i - 1];
    }

    // thrust::device_vector<tuple_type> d_recv_buffer(total_recv * arity);
    // use cuda malloc to allocate the memory for the receive buffer
    column_type *recv_buffer;
    checkCuda(
        cudaMalloc(&recv_buffer, total_recv * arity * sizeof(column_type)));

    // send the tuples to the other ranks
    if (gpu_direct_flag) {
        MPI_Alltoallv(d_send_buffer.data().get(),
                      h_rank_tuple_send_counts.data(),
                      send_displacements.data(), MPI_ELEM_TYPE, recv_buffer,
                      h_rank_tuple_recv_counts.data(),
                      recv_displacements.data(), MPI_ELEM_TYPE, MPI_COMM_WORLD);
    } else {
        if (rank == 0) {
            std::cout << "Warnning using host memory for MPI_Alltoallv, GPU "
                         "directe disabled"
                      << std::endl;
        }
        thrust::host_vector<column_type> h_send_buffer(d_send_buffer);
        thrust::host_vector<column_type> h_recv_buffer(total_recv * arity);
        MPI_Alltoallv(h_send_buffer.data(), h_rank_tuple_send_counts.data(),
                      send_displacements.data(), MPI_ELEM_TYPE,
                      h_recv_buffer.data(), h_rank_tuple_recv_counts.data(),
                      recv_displacements.data(), MPI_ELEM_TYPE, MPI_COMM_WORLD);
        cudaMemcpy(recv_buffer, h_recv_buffer.data(),
                   total_recv * arity * sizeof(column_type),
                   cudaMemcpyHostToDevice);
    }

    // container
    container->reload(recv_buffer, total_recv);
    container->sort();
    container->dedup();
}

tuple_size_t
Communicator::gatherRelContainerSize(GHashRelContainer *container) {
    if (total_rank == 1) {
        return container->tuple_counts;
    }
    tuple_size_t total_size = 0;
    MPI_Allreduce(&container->tuple_counts, &total_size, 1, MPI_ELEM_TYPE,
                  MPI_SUM, MPI_COMM_WORLD);
    return total_size;
}

bool Communicator::reduceBool(bool value) {
    if (total_rank == 1) {
        return value;
    }
    int result = 0;
    MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    return result;
}

tuple_size_t Communicator::reduceSumTupleSize(tuple_size_t value) {
    if (total_rank == 1) {
        return value;
    }
    tuple_size_t result = 0;
    MPI_Allreduce(&value, &result, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    return result;
}

Communicator::~Communicator() {
    // Finalize the MPI environment
    if (is_initialized) {
        MPI_Finalize();
    }
}

void Communicator::computeSubBucket(
    GHashRelContainer *rel_container, bucket_map_t &sub_bucket_map,
    thrust::device_vector<bucket_id_t> &tuple_subbucket_map) {
    // compute the subbucket for each tuple in the relation
    auto tuple_counts = rel_container->tuple_counts;

    tuple_subbucket_map.resize(tuple_counts);

    auto ni_size = rel_container->arity - rel_container->index_column_size;
    auto i_size = rel_container->index_column_size;
    auto sub_bucket_size = sub_bucket_map.size();

    thrust::transform(
        thrust::device, rel_container->tuples,
        rel_container->tuples + tuple_counts, tuple_subbucket_map.begin(),
        [rel_container, ni_size, i_size,
         sub_bucket_size] __device__(const tuple_type &tuple) -> bucket_id_t {
            return (bucket_id_t)(prefix_hash(tuple + i_size, ni_size) %
                                 sub_bucket_size);
        });
}

void Communicator::computeNewSubBucketSize(Relation *rel) {
    tuple_size_t total_size_full = 0;

    for (int b_id = 0; b_id < rel->sub_bucket_size; b_id++) {
        total_size_full += rel->fulls[b_id]->tuple_counts;
    }

    thrust::host_vector<bucket_id_t> full_size_all_ranks;

    // MPI all to all communication to write the total full_size
    // on each rank into full_size_all_ranks.

    full_size_all_ranks.resize(total_rank);
    MPI_Allgather(&total_size_full, 1, MPI_UINT64_T, full_size_all_ranks.data(),
                  1, MPI_UINT64_T, MPI_COMM_WORLD);

    // find the smallest full size
    auto min_full_size = *thrust::min_element(full_size_all_ranks.begin(),
                                              full_size_all_ranks.end());

    // check if current_rank size is "threshold" time larger than smallest size
    // if so means we need rebalance
    bool need_rebalance =
        (min_full_size * rebalance_threshold < total_size_full);

    if (!need_rebalance) {
        return;
    }

    rel->sub_bucket_size = rel->sub_bucket_size * split_constant;
}

void Communicator::computeSubBucketMap(Relation *rel) {
    auto old_size = rel->sub_bucket_size;
    computeNewSubBucketSize(rel);
    auto new_size = rel->sub_bucket_size;
    if (old_size == new_size) {
        return;
    }

    // round robin assign the sub bucket
    // which mean subbucket 0 is current rank, 1 is current rank + 1, etc
    for (size_t i = 0; i < rel->sub_bucket_size; i++) {
        rel->sub_bucket_map[i] = (rank + i) % total_rank;
    }
}

void Communicator::rebalance(Relation *rel) {
    // compute the new sub bucket size
    computeNewSubBucketSize(rel);
    // compute the new sub bucket map
    computeSubBucketMap(rel);
    // distribute everything
    for (int i = 0; i < rel->sub_bucket_size; i++) {
        thrust::device_vector<bucket_id_t> tuple_subbucket_map;
        computeSubBucket(rel->fulls[i], rel->sub_bucket_map,
                         tuple_subbucket_map);
        // TODO: distribute by subbucket
    }
}

void map_to_device_vec(bucket_map_t &bucket_map,
                       thrust::device_vector<uint8_t> &d_bucket_map) {
    thrust::host_vector<uint8_t> tmp(bucket_map.size());
    for (auto &p : bucket_map) {
        tmp[p.first] = p.second;
    }
    d_bucket_map = tmp;
}

void Communicator::distribute_by_rank_mapping(
    GHashRelContainer *container, bucket_map_t &bucket_map,
    thrust::device_vector<bucket_id_t> &tuple_bucket_map) {
    //
    if (total_rank == 1) {
        return;
    }
    auto arity = container->arity;

    thrust::device_vector<uint8_t> bucket_rank_map_d(bucket_map.size());
    map_to_device_vec(bucket_map, bucket_rank_map_d);

    // get actual rank of tuple_bucket_map based on on bucket_map
    // thrust::transform(
    //     thrust::device, tuple_bucket_map.begin(), tuple_bucket_map.end(),
    //     tuple_bucket_map.begin(),
    //     [map = bucket_rank_map_d.data().get()] __device__(
    //         const bucket_id_t &b_id) -> bucket_id_t { return map[b_id]; });
    thrust::gather(thrust::device, tuple_bucket_map.begin(),
                   tuple_bucket_map.end(), bucket_rank_map_d.begin(),
                   tuple_bucket_map.begin());

    // stable sort the tuples based on the rank
    thrust::stable_sort_by_key(thrust::device, tuple_bucket_map.begin(),
                               tuple_bucket_map.end(), container->tuples);
    // tuple size need to send to each rank
    thrust::device_vector<int> rank_tuple_counts(total_rank);
    thrust::device_vector<uint8_t> reduced_rank(total_rank);

    auto reduced_end = thrust::reduce_by_key(
        thrust::device, tuple_bucket_map.begin(), tuple_bucket_map.end(),
        thrust::constant_iterator<int>(1), reduced_rank.begin(),
        rank_tuple_counts.begin());
    auto rank_tuple_counts_size = reduced_end.first - reduced_rank.begin();
    rank_tuple_counts.resize(rank_tuple_counts_size);
    reduced_rank.resize(rank_tuple_counts_size);
    // create a host copy of the rank tuple counts and reduced rank
    thrust::host_vector<int> h_rank_tuple_counts(rank_tuple_counts);
    thrust::host_vector<uint8_t> h_reduced_rank(reduced_rank);

    thrust::host_vector<int> h_rank_tuple_send_counts(total_rank);
    for (int i = 0; i < rank_tuple_counts_size; i++) {
        h_rank_tuple_send_counts[h_reduced_rank[i]] = h_rank_tuple_counts[i];
    }
    thrust::host_vector<int> h_rank_tuple_recv_counts(total_rank);

    int total_send = thrust::reduce(h_rank_tuple_send_counts.begin(),
                                    h_rank_tuple_send_counts.end());

    // send the tuple size to each rank
    MPI_Alltoall(h_rank_tuple_send_counts.data(), 1, MPI_INT,
                 h_rank_tuple_recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total_recv = thrust::reduce(h_rank_tuple_recv_counts.begin(),
                                    h_rank_tuple_recv_counts.end());

    // allocate memory for the send and receive buffers
    thrust::device_vector<column_type> d_send_buffer(total_send * arity);

    // copy tuples to the send buffer
    thrust::for_each(
        thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(
            container->tuples, thrust::counting_iterator<uint32_t>(0))),
        thrust::make_zip_iterator(thrust::make_tuple(
            container->tuples + container->tuple_counts,
            thrust::counting_iterator<uint32_t>(container->tuple_counts))),
        [dest = d_send_buffer.data().get(), arity] __device__(
            const thrust::tuple<tuple_type, uint32_t> &t) -> void {
            auto &tuple = thrust::get<0>(t);
            auto &index = thrust::get<1>(t);
            auto dest_tp = dest + index * arity;
            for (int i = 0; i < arity; i++) {
                dest_tp[i] = tuple[i];
            }
        });

    // print send buffer on each rank after copy

    // after send, free the memory  of the container
    free_relation_container(container);
    container->tuple_counts = total_recv;

    // convert the tuple size to column size by times arity on each element
    // in h_rank_tuple_(send/recv)_counts
    thrust::transform(
        h_rank_tuple_send_counts.begin(), h_rank_tuple_send_counts.end(),
        thrust::make_constant_iterator(arity), h_rank_tuple_send_counts.begin(),
        thrust::multiplies<uint32_t>());
    thrust::transform(
        h_rank_tuple_recv_counts.begin(), h_rank_tuple_recv_counts.end(),
        thrust::make_constant_iterator(arity), h_rank_tuple_recv_counts.begin(),
        thrust::multiplies<uint32_t>());

    // create displacements for the send and receive buffers
    thrust::host_vector<int> send_displacements(total_rank);
    thrust::host_vector<int> recv_displacements(total_rank);
    send_displacements[0] = 0;
    recv_displacements[0] = 0;
    for (int i = 1; i < total_rank; i++) {
        send_displacements[i] =
            send_displacements[i - 1] + h_rank_tuple_send_counts[i - 1];
        recv_displacements[i] =
            recv_displacements[i - 1] + h_rank_tuple_recv_counts[i - 1];
    }

    // thrust::device_vector<tuple_type> d_recv_buffer(total_recv * arity);
    // use cuda malloc to allocate the memory for the receive buffer
    column_type *recv_buffer;
    checkCuda(
        cudaMalloc(&recv_buffer, total_recv * arity * sizeof(column_type)));

    // send the tuples to the other ranks
    if (gpu_direct_flag) {
        MPI_Alltoallv(d_send_buffer.data().get(),
                      h_rank_tuple_send_counts.data(),
                      send_displacements.data(), MPI_ELEM_TYPE, recv_buffer,
                      h_rank_tuple_recv_counts.data(),
                      recv_displacements.data(), MPI_ELEM_TYPE, MPI_COMM_WORLD);
    } else {
        if (rank == 0) {
            std::cout << "Warnning using host memory for MPI_Alltoallv, GPU "
                         "directe disabled"
                      << std::endl;
        }
        thrust::host_vector<column_type> h_send_buffer(d_send_buffer);
        thrust::host_vector<column_type> h_recv_buffer(total_recv * arity);
        MPI_Alltoallv(h_send_buffer.data(), h_rank_tuple_send_counts.data(),
                      send_displacements.data(), MPI_ELEM_TYPE,
                      h_recv_buffer.data(), h_rank_tuple_recv_counts.data(),
                      recv_displacements.data(), MPI_ELEM_TYPE, MPI_COMM_WORLD);
        cudaMemcpy(recv_buffer, h_recv_buffer.data(),
                   total_recv * arity * sizeof(column_type),
                   cudaMemcpyHostToDevice);
    }

    // container
    container->reload(recv_buffer, total_recv);
    container->sort();
    container->dedup();
}
