
#include "../include/comm.h"
#include "../include/exception.cuh"
#include "../include/print.cuh"
#include "../include/timer.cuh"
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#include <rmm/exec_policy.hpp>
#include <rmm/device_vector.hpp>

void Communicator::init(int argc, char **argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &total_rank);
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    is_initialized = true;
}

void Communicator::distribute(GHashRelContainer *container) {
    // Distribute the data
    // compute the rank of tuple in the container
    if (total_rank == 1) {
        return;
    }
    auto arity = container->arity;

    KernelTimer timer;

    // tuple_rank_mapping.clear();
    tuple_rank_mapping.resize(container->tuple_counts);

    timer.start_timer();
    thrust::transform(
        thrust::device, container->tuples,
        container->tuples + container->tuple_counts, tuple_rank_mapping.begin(),
        [total_rank = total_rank, jc = container->index_column_size] __device__(
            const tuple_type &tuple) -> uint8_t {
            return (uint8_t)(prefix_hash(tuple, jc) % total_rank);
        });
    timer.stop_timer();
    time_detail[0] += timer.get_spent_time();

    // stable sort the tuples based on the rank
    timer.start_timer();
    thrust::stable_sort_by_key(rmm::exec_policy(), tuple_rank_mapping.begin(),
                               tuple_rank_mapping.end(), container->tuples);
    timer.stop_timer();
    time_detail[1] += timer.get_spent_time();
    // tuple size need to send to each rank
    rmm::device_vector<int> rank_tuple_counts(total_rank);
    rmm::device_vector<uint8_t> reduced_rank(total_rank);

    timer.start_timer();
    auto reduced_end = thrust::reduce_by_key(
        thrust::device, tuple_rank_mapping.begin(), tuple_rank_mapping.end(),
        thrust::constant_iterator<int>(1), reduced_rank.begin(),
        rank_tuple_counts.begin());
    auto rank_tuple_counts_size = reduced_end.first - reduced_rank.begin();
    timer.stop_timer();
    time_detail[2] += timer.get_spent_time();
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

    timer.start_timer();
    int total_send = thrust::reduce(h_rank_tuple_send_counts.begin(),
                                    h_rank_tuple_send_counts.end());

    // send the tuple size to each rank
    MPI_Alltoall(h_rank_tuple_send_counts.data(), 1, MPI_INT,
                 h_rank_tuple_recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total_recv = thrust::reduce(h_rank_tuple_recv_counts.begin(),
                                    h_rank_tuple_recv_counts.end());
    timer.stop_timer();
    time_detail[3] += timer.get_spent_time();


    timer.start_timer();
    // allocate memory for the send and receive buffers
    rmm::device_vector<column_type> d_send_buffer(total_send * arity);

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
    timer.stop_timer();
    time_detail[4] += timer.get_spent_time();

    // print send buffer on each rank after copy

    // after send, free the memory  of the container
    container->free();
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

    // rmm::device_vector<tuple_type> d_recv_buffer(total_recv * arity);
    // use cuda malloc to allocate the memory for the receive buffer
    timer.start_timer();
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
    timer.stop_timer();
    time_detail[5] += timer.get_spent_time();

    // container
    container->reload(recv_buffer, total_recv);

    timer.start_timer();
    container->sort();
    timer.stop_timer();
    time_detail[6] += timer.get_spent_time();
    timer.start_timer();
    container->dedup();
    timer.stop_timer();
    time_detail[7] += timer.get_spent_time();
}

void Communicator::broadcast(GHashRelContainer *container) {
    if (total_rank == 1) {
        return;
    }
    container->fit();

    thrust::host_vector<int> h_rank_tuple_send_counts(total_rank);
    thrust::host_vector<int> h_rank_tuple_recv_counts(total_rank);
    thrust::fill(h_rank_tuple_send_counts.begin(),
                 h_rank_tuple_send_counts.end(),
                 container->tuple_counts * container->arity);
    MPI_Alltoall(h_rank_tuple_send_counts.data(), 1, MPI_INT,
                 h_rank_tuple_recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total_recv = thrust::reduce(h_rank_tuple_recv_counts.begin(),
                                    h_rank_tuple_recv_counts.end());
    int total_send = thrust::reduce(h_rank_tuple_send_counts.begin(),
                                    h_rank_tuple_send_counts.end());

    // print h_rank_tuple_recv_counts

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

    // use cuda malloc to allocate the memory for the receive buffer
    column_type *recv_buffer;
    checkCuda(cudaMalloc(&recv_buffer, total_recv * sizeof(column_type)));

    // prepare the device send buffer
    rmm::device_vector<column_type> d_send_buffer(total_send);
    for (int i = 0; i < total_rank; i++) {
        cudaMemcpy(d_send_buffer.data().get() +
                       i * container->data_raw_row_size * container->arity,
                   container->data_raw,
                   container->data_raw_row_size * container->arity *
                       sizeof(column_type),
                   cudaMemcpyDeviceToDevice);
    }

    // send the tuples to the other ranks
    if (gpu_direct_flag) {
        MPI_Alltoallv(container->tuples, h_rank_tuple_send_counts.data(),
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
        thrust::host_vector<column_type> h_recv_buffer(total_recv);
        MPI_Alltoallv(h_send_buffer.data(), h_rank_tuple_send_counts.data(),
                      send_displacements.data(), MPI_ELEM_TYPE,
                      h_recv_buffer.data(), h_rank_tuple_recv_counts.data(),
                      recv_displacements.data(), MPI_ELEM_TYPE, MPI_COMM_WORLD);
        cudaMemcpy(recv_buffer, h_recv_buffer.data(),
                   total_recv * sizeof(column_type), cudaMemcpyHostToDevice);
    }

    container->free();
    // container
    container->reload(recv_buffer, total_recv / container->arity);
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
    MPI_Allreduce(&value, &result, 1, MPI_ELEM_TYPE, MPI_SUM, MPI_COMM_WORLD);
    return result;
}

Communicator::~Communicator() {
    // Finalize the MPI environment
    if (is_initialized) {
        MPI_Finalize();
    }
}
