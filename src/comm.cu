
#include "../include/comm.h"
#include "../include/exception.cuh"
#include "../include/print.cuh"
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
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
}

void Communicator::distribute(GHashRelContainer *container) {
    // Distribute the data  w
    // compute the rank of tuple in the container
    auto arity = container->arity;

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
    thrust::device_vector<uint32_t> rank_tuple_counts(total_rank);

    thrust::reduce_by_key(
        thrust::device, tuple_rank_mapping.begin(), tuple_rank_mapping.end(),
        thrust::constant_iterator<uint32_t>(1), thrust::make_discard_iterator(),
        rank_tuple_counts.begin());

    // copy the tuple size to the host
    thrust::host_vector<int> h_rank_tuple_send_counts(rank_tuple_counts);
    thrust::host_vector<int> h_rank_tuple_recv_counts(total_rank);

    int total_send = thrust::reduce(h_rank_tuple_send_counts.begin(),
                                    h_rank_tuple_send_counts.end());

    // send the tuple size to each rank
    MPI_Alltoall(h_rank_tuple_send_counts.data(), 1, MPI_INT,
                 h_rank_tuple_recv_counts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    // recive displacements
    int total_recv = thrust::reduce(h_rank_tuple_recv_counts.begin(),
                                    h_rank_tuple_recv_counts.end());

    // allocate memory for the send and receive buffers
    thrust::device_vector<column_type> d_send_buffer(total_send * arity);
    // thrust::device_vector<tuple_type> d_recv_buffer(total_recv * arity);
    // use cuda malloc to allocate the memory for the receive buffer
    column_type *recv_buffer;
    checkCuda(
        cudaMalloc(&recv_buffer, total_recv * arity * sizeof(column_type)));

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

    //
    thrust::host_vector<column_type> h_send_buffer(d_send_buffer);
    // thrust::host_vector<column_type> h_recv_buffer(total_recv * arity);
    // std::cout << "rank " << rank << " total recv count " << total_recv << " before alltoallv\n";

    // MPI_Alltoallv(h_send_buffer.data(), h_rank_tuple_send_counts.data(),
    //               send_displacements.data(), MPI_ELEM_TYPE, h_recv_buffer.data(),
    //               h_rank_tuple_recv_counts.data(), recv_displacements.data(),
    //               MPI_ELEM_TYPE, MPI_COMM_WORLD);

    //

    // send the tuples to the other ranks
    MPI_Alltoallv(d_send_buffer.data().get(), h_rank_tuple_send_counts.data(),
                  send_displacements.data(), MPI_INT, recv_buffer,
                  h_rank_tuple_recv_counts.data(), recv_displacements.data(),
                  MPI_INT, MPI_COMM_WORLD);

    // cudaMemcpy(recv_buffer, h_recv_buffer.data(), total_recv * arity * sizeof(column_type), cudaMemcpyHostToDevice);
    // swap recv_buffer with the container
    container->data_raw = recv_buffer;
    container->data_raw_row_size = total_recv;
    // reload the container
    checkCuda(cudaMalloc(&container->tuples, total_recv * sizeof(tuple_type)));
    thrust::transform(thrust::device, thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(total_recv),
                      container->tuples,
                      [raw_ptr = container->data_raw, arity] __device__(
                          int i) -> tuple_type { return raw_ptr + i * arity; });
    thrust::sort(thrust::device, container->tuples,
                 container->tuples + total_recv,
                 tuple_indexed_less(container->index_column_size, arity));
    auto new_end =
        thrust::unique(thrust::device, container->tuples,
                       container->tuples + total_recv, t_equal(arity));
    container->tuple_counts = new_end - container->tuples;
}

Communicator::~Communicator() {
    // Finalize the MPI environment
    MPI_Finalize();
}