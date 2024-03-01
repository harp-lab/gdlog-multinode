
#include "../../include/print.cuh"
#include "../../include/relational_algebra.cuh"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

//  a gpu kernel to do cartesian product on two relations
//  2 phase, get size of result, then do the actual product
__global__ void cartesian_product_size(column_type *d_a, column_type *d_b,
                                       tuple_size_t *d_size,
                                       tuple_size_t a_rows, int a_cols,
                                       tuple_size_t b_rows, int b_cols,
                                       TupleJoinFilter tp_filter) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= a_rows)
        return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < a_rows; i += stride) {
        unsigned long long count = 0;
        // store the current tuple of a local
        column_type a_local[MAX_ARITY];
        for (int j = 0; j < a_cols; j++) {
            a_local[j] = d_a[i * a_cols + j];
        }
        for (int j = 0; j < b_rows; j++) {
            tuple_type cur_b_tp = d_b + j * b_cols;
            if (tp_filter(a_local, cur_b_tp)) {
                count++;
            }
        }
        d_size[i] = count;
    }
}

__global__ void
cartesian_product(column_type *d_a, column_type *d_b, column_type *d_result,
                  tuple_size_t *d_result_offset, tuple_size_t a_rows,
                  int a_cols, tuple_size_t b_rows, int b_cols,
                  TupleGenerator tp_gen, TupleJoinFilter tp_filter) {
    int out_arity = tp_gen.arity;
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= a_rows)
        return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < a_rows; i += stride) {
        unsigned long long count = 0;
        tuple_size_t offset = d_result_offset[i];
        // store the current tuple of a local
        column_type a_local[MAX_ARITY];
        for (int j = 0; j < a_cols; j++) {
            a_local[j] = d_a[i * a_cols + j];
        }
        for (int j = 0; j < b_rows; j++) {
            tuple_type cur_b_tp = d_b + j * b_cols;
            tuple_type out_tuple = d_result + (offset + count) * out_arity;
            if (tp_filter(a_local, cur_b_tp)) {
                tp_gen(a_local, cur_b_tp, out_tuple);
                count++;
            }
        }
    }
}

void RelationalCartesian::operator()() {
    GHashRelContainer *inner;
    if (inner_ver == DELTA) {
        inner = inner_rel->delta;
    } else if (inner_ver == FULL) {
        inner = inner_rel->full;
    } else {
        inner = inner_rel->newt;
    }
    GHashRelContainer *outer;
    if (outer_ver == DELTA) {
        outer = outer_rel->delta;
    } else if (outer_ver == FULL) {
        outer = outer_rel->full;
    } else {
        outer = outer_rel->newt;
    }
    int output_arity = output_rel->arity;
    // GHashRelContainer* output = output_rel->newt;

    if (outer->tuples == nullptr || outer->tuple_counts == 0) {
        output_rel->newt->tuple_counts = 0;
        return;
    }
    if (inner->tuples == nullptr || inner->tuple_counts == 0) {
        output_rel->newt->tuple_counts = 0;
        return;
    }

    // std::cout << "outer: " << outer->tuple_counts << " inner: " << inner->tuple_counts << std::endl;
    // inner_rel->defragement(inner_ver);
    // print_tuple_rows(outer, "outer");
    // get the size of the result
    tuple_size_t *d_size;
    cudaMalloc((void **)&d_size, outer->tuple_counts * sizeof(tuple_size_t));
    cartesian_product_size<<<grid_size, block_size>>>(
        outer->data_raw, inner->data_raw, d_size, outer->tuple_counts,
        outer->arity, inner->tuple_counts, inner->arity, tuple_pred);
    cudaDeviceSynchronize();
    // get total size of the result by reduce
    tuple_size_t result_size =
        thrust::reduce(thrust::device, d_size, d_size + outer->tuple_counts);
    
    std::cout << "cartesian result size: " << result_size << std::endl;
    // print d_size
    thrust::host_vector<tuple_size_t> h_size(inner->tuple_counts);
    cudaMemcpy(h_size.data(), d_size, inner->tuple_counts * sizeof(tuple_size_t),
               cudaMemcpyDeviceToHost);
    // for (int i = 0; i < inner->tuple_counts; i++) {
    //     std::cout << h_size[i] << " ";
    // }

    // get the offset of each tuple
    thrust::exclusive_scan(thrust::device, d_size, d_size + outer->tuple_counts,
                           d_size);
    // allocate memory for the result
    column_type *d_result;
    cudaMalloc((void **)&d_result,
               result_size * output_rel->arity * sizeof(column_type));
    // do the actual product
    cartesian_product<<<grid_size, block_size>>>(
        outer->data_raw, inner->data_raw, d_result, d_size, outer->tuple_counts,
        outer->arity, inner->tuple_counts, inner->arity, tuple_generator,
        tuple_pred);
    cudaDeviceSynchronize();
    // free the memory
    cudaFree(d_size);
    // set the result
    if (output_rel->newt->tuple_counts == 0) {
        output_rel->newt->reload(d_result, result_size);
        // print_tuple_rows(output_rel->newt, "newt");
    } else {
        GHashRelContainer *temp = new GHashRelContainer(
            output_rel->arity, output_rel->index_column_size, 0);
        temp->tuple_counts = result_size;
        temp->arity = output_arity;
        temp->reload(d_result, result_size);
        temp->sort();
        RelationalUnion ru(temp ,output_rel->newt);
        ru();
        output_rel->newt->fit();
        temp->free();
        delete temp;
    }
}
