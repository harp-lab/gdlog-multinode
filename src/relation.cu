
#include "../include/exception.cuh"
#include "../include/print.cuh"
#include "../include/relation.cuh"
#include "../include/timer.cuh"
#include "../include/tuple.cuh"
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/merge.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

__global__ void calculate_index_hash(GHashRelContainer *target,
                                     tuple_indexed_less cmp) {
    tuple_size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= target->tuple_counts)
        return;

    tuple_size_t stride = blockDim.x * gridDim.x;

    for (tuple_size_t i = index; i < target->tuple_counts; i += stride) {
        tuple_type cur_tuple = target->tuples[i];

        u64 hash_val = prefix_hash(cur_tuple, target->index_column_size);
        u64 request_hash_index = hash_val % target->index_map_size;
        tuple_size_t position = request_hash_index;
        // insert into data container
        while (true) {
            // critical condition!
            u64 existing_key = atomicCAS(&(target->index_map[position].key),
                                         EMPTY_HASH_ENTRY, hash_val);
            auto existing_value = target->index_map[position].value;
            if (existing_key == EMPTY_HASH_ENTRY || existing_key == hash_val) {
                bool collison_flag = false;
                while (true) {
                    if (existing_value < i) {
                        // occupied entry, but no need for swap, just check if
                        // collision
                        if (!tuple_eq(target->tuples[existing_value], cur_tuple,
                                      target->index_column_size)) {
                            // collision, find nex available entry
                            collison_flag = true;
                            break;
                        } else {
                            // no collision but existing tuple is smaller, in
                            // this case, not need to swap, just return(break;
                            // break)
                            break;
                        }
                    }
                    if (existing_value > i &&
                        existing_value != EMPTY_HASH_ENTRY) {
                        // occupied entry, may need for swap
                        if (!tuple_eq(target->tuples[existing_value], cur_tuple,
                                      target->index_column_size)) {
                            // collision, find nex available entry
                            collison_flag = true;
                            break;
                        }
                        // else, swap
                    }
                    // swap value
                    if (existing_value == i) {
                        // swap success return
                        break;
                    } else {
                        // need swap
                        existing_value =
                            atomicCAS(&(target->index_map[position].value),
                                      existing_value, (unsigned long long)i);
                    }
                }
                if (!collison_flag) {
                    break;
                }
            }

            position = (position + 1) % target->index_map_size;
        }
    }
}

// template <typename tp_gen_t>

__global__ void get_copy_result(tuple_type *src_tuples,
                                column_type *dest_raw_data, int output_arity,
                                tuple_size_t tuple_counts,
                                TupleProjector tp_gen) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= tuple_counts)
        return;

    int stride = blockDim.x * gridDim.x;
    for (tuple_size_t i = index; i < tuple_counts; i += stride) {
        tuple_type dest_tp = dest_raw_data + output_arity * i;
        tp_gen(src_tuples[i], dest_tp);
    }
}

void Relation::flush_delta(int grid_size, int block_size, float *detail_time) {
    if (delta->tuple_counts == 0) {
        return;
    }
    KernelTimer timer;
    timer.start_timer();
    tuple_type *tuple_full_buf;
    tuple_size_t new_full_size = full->tuple_counts + delta->tuple_counts;

    bool extened_mem = false;

    tuple_size_t total_mem_size = get_total_memory();
    tuple_size_t free_mem = get_free_memory();
    u64 delta_mem_size = delta->tuple_counts * sizeof(tuple_type);
    int multiplier = FULL_BUFFER_VEC_MULTIPLIER;
    if (!pre_allocated_merge_buffer_flag && !fully_disable_merge_buffer_flag &&
        delta_mem_size * multiplier <= 0.1 * free_mem) {
        std::cout << "reenable pre-allocated merge buffer" << std::endl;
        pre_allocated_merge_buffer_flag = true;
    }

    if (!fully_disable_merge_buffer_flag && pre_allocated_merge_buffer_flag) {
        tuple_merge_buffer.resize(full->tuple_counts +
                                  delta->tuple_counts * multiplier);
        tuple_merge_buffer.shrink_to_fit();
        tuple_merge_buffer_size =
            full->tuple_counts + delta->tuple_counts * multiplier;
        tuple_full_buf = tuple_merge_buffer.data().get();
    } else {
        tuple_merge_buffer_size = full->tuple_counts + delta->tuple_counts;
        tuple_merge_buffer.resize(tuple_merge_buffer_size);
        tuple_merge_buffer.shrink_to_fit();
        tuple_full_buf = tuple_merge_buffer.data().get();
        // checkCuda(cudaMemset(tuple_full_buf, 0, tuple_full_buf_mem_size));
        // checkCuda(cudaDeviceSynchronize());
    }
    // std::cout << new_full_size << std::endl;

    timer.stop_timer();
    // std::cout << "malloc time : " << timer.get_spent_time() << std::endl;
    detail_time[0] = timer.get_spent_time();

    // get current rank
    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // if (rank == 1) {
    //     std::cout << ">>>>>>>>> rank 1 : "
    //               << " delta size : " << delta->tuple_counts
    //               << " delta arity : " << delta->arity
    //               << " delta index column size : " <<
    //               delta->index_column_size
    //               << std::endl;
    // }
    timer.start_timer();
    tuple_type *end_tuple_full_buf = thrust::merge(
        thrust::device, full->tuples, full->tuples + full->tuple_counts,
        delta->tuples, delta->tuples + delta->tuple_counts, tuple_full_buf,
        tuple_indexed_less(delta->index_column_size, delta->arity));
    timer.stop_timer();
    // std::cout << "merge time : " << timer.get_spent_time() << std::endl;
    detail_time[1] = timer.get_spent_time();
    // checkCuda(cudaDeviceSynchronize());
    full->tuple_counts = new_full_size;

    timer.start_timer();
    if (!fully_disable_merge_buffer_flag && pre_allocated_merge_buffer_flag) {
        auto &old_full = full->tuples_vec;
        full->tuples = tuple_merge_buffer.data().get();
        tuple_merge_buffer.swap(old_full);
        if (extened_mem) {
            tuple_merge_buffer.resize(new_full_size);
        }
    } else {
        full->tuples_vec.swap(tuple_merge_buffer);
        full->tuples = full->tuples_vec.data().get();
        tuple_merge_buffer.clear();
        tuple_merge_buffer.shrink_to_fit();
    }
    timer.stop_timer();
    detail_time[2] = timer.get_spent_time();
    buffered_delta_vectors.push_back(delta);
    if (index_flag) {
        full->build_index(grid_size, block_size);
    }
}

void load_relation_container(GHashRelContainer *target, int arity,
                             column_type *data, tuple_size_t data_row_size,
                             tuple_size_t index_column_size,
                             int dependent_column_size,
                             float index_map_load_factor, int grid_size,
                             int block_size, float *detail_time,
                             bool gpu_data_flag, bool sorted_flag,
                             bool build_index_flag, bool tuples_array_flag) {
    KernelTimer timer;
    target->arity = arity;
    target->tuple_counts = data_row_size;
    target->data_raw_row_size = data_row_size;
    target->index_map_load_factor = index_map_load_factor;
    target->index_column_size = index_column_size;
    target->dependent_column_size = dependent_column_size;
    // load index selection into gpu
    // u64 index_columns_mem_size = index_column_size * sizeof(u64);
    // checkCuda(cudaMalloc((void**) &(target->index_columns),
    // index_columns_mem_size)); cudaMemcpy(target->index_columns,
    // index_columns, index_columns_mem_size, cudaMemcpyHostToDevice);
    if (data_row_size == 0) {
        return;
    }
    // load raw data from host
    if (gpu_data_flag) {
        target->data_raw = data;
    } else {
        u64 relation_mem_size =
            data_row_size * ((u64)arity) * sizeof(column_type);
        checkCuda(cudaMalloc((void **)&(target->data_raw), relation_mem_size));
        checkCuda(cudaMemcpy(target->data_raw, data, relation_mem_size,
                             cudaMemcpyHostToDevice));
    }
    if (tuples_array_flag) {
        // init tuple to be unsorted raw tuple data address
        target->reload(target->data_raw, data_row_size);
    }
    // sort raw data
    if (!sorted_flag) {
        timer.start_timer();
        target->sort();
        // print_tuple_rows(target, "after sort");
        timer.stop_timer();
        detail_time[0] = timer.get_spent_time();
        // deduplication here?
        timer.start_timer();
        target->dedup();
        timer.stop_timer();
        detail_time[1] = timer.get_spent_time();
    }

    if (build_index_flag) {
        timer.start_timer();
        target->build_index(grid_size, block_size);
        timer.stop_timer();
        detail_time[2] = timer.get_spent_time();
    }
}

void repartition_relation_index(GHashRelContainer *target, int arity,
                                column_type *data, tuple_size_t data_row_size,
                                tuple_size_t index_column_size,
                                int dependent_column_size,
                                float index_map_load_factor, int grid_size,
                                int block_size, float *detail_time) {
    KernelTimer timer;
    target->arity = arity;
    target->tuple_counts = data_row_size;
    target->data_raw_row_size = data_row_size;
    target->index_map_load_factor = index_map_load_factor;
    target->index_column_size = index_column_size;
    target->dependent_column_size = dependent_column_size;
    if (data_row_size == 0) {
        return;
    }
    target->reload(data, data_row_size);

    timer.start_timer();
    thrust::sort(thrust::device, target->tuples, target->tuples + data_row_size,
                 tuple_indexed_less(index_column_size, arity));
    // print_tuple_rows(target, "after sort");
    timer.stop_timer();
    detail_time[0] = timer.get_spent_time();
    detail_time[1] = timer.get_spent_time();

    target->tuple_counts = data_row_size;
    // print_tuple_rows(target, "after dedup");

    timer.start_timer();
    target->build_index(grid_size, block_size);
    timer.stop_timer();
    detail_time[2] = timer.get_spent_time();
}

void reload_full_temp(GHashRelContainer *target, int arity, tuple_type *tuples,
                      tuple_size_t data_row_size,
                      tuple_size_t index_column_size, int dependent_column_size,
                      float index_map_load_factor, int grid_size,
                      int block_size) {
    //
    target->arity = arity;
    target->tuple_counts = data_row_size;
    target->index_map_load_factor = index_map_load_factor;
    target->index_column_size = index_column_size;
    target->dependent_column_size = dependent_column_size;
    target->tuples = tuples;
    target->index_map_size = std::ceil(data_row_size / index_map_load_factor);
    // target->index_map_size = data_row_size;
    target->build_index(grid_size, block_size);
}

void GHashRelContainer::free() {
    tuple_counts = 0;
    index_map_size = 0;
    data_raw_row_size = 0;
    if (index_map != nullptr) {
        checkCuda(cudaFree(index_map));
        index_map = nullptr;
    }
    if (tuples != nullptr) {
        // checkCuda(cudaFree(tuples));
        tuples_vec.clear();
        tuples_vec.shrink_to_fit();
        tuples = nullptr;
    }
    if (data_raw != nullptr) {
        checkCuda(cudaFree(data_raw));
        data_raw = nullptr;
    }
}

void load_relation(Relation *target, std::string name, int arity,
                   column_type *data, tuple_size_t data_row_size,
                   tuple_size_t index_column_size, int dependent_column_size,
                   int grid_size, int block_size, bool tmp_flag) {

    target->name = name;
    target->arity = arity;
    target->index_column_size = index_column_size;
    target->dependent_column_size = dependent_column_size;
    target->tmp_flag = tmp_flag;
    target->full =
        new GHashRelContainer(arity, index_column_size, dependent_column_size);
    target->delta =
        new GHashRelContainer(arity, index_column_size, dependent_column_size);
    target->newt =
        new GHashRelContainer(arity, index_column_size, dependent_column_size);
    // target->newt->tmp_flag = tmp_flag;

    float detail_time[5];
    // everything must have a full
    load_relation_container(target->full, arity, data, data_row_size,
                            index_column_size, dependent_column_size, 0.8,
                            grid_size, block_size, detail_time);
}

void GHashRelContainer::sort() {
    thrust::sort(thrust::device, this->tuples,
                 this->tuples + this->tuple_counts,
                 tuple_indexed_less(this->index_column_size, arity));
}

void GHashRelContainer::dedup() {
    tuple_type *new_end =
        thrust::unique(thrust::device, this->tuples,
                       this->tuples + this->tuple_counts, t_equal(this->arity));
    this->tuple_counts = new_end - this->tuples;
}

void GHashRelContainer::reload(column_type *data, tuple_size_t data_row_size) {
    if (this->data_raw != nullptr && this->data_raw != data) {
        checkCuda(cudaFree(this->data_raw));
    }
    data_raw = data;
    this->data_raw_row_size = data_row_size;
    // if (this->tuples != nullptr) {
    //     // checkCuda(cudaFree(this->tuples));
    //     tuples_vec.clear();
    // }

    tuples_vec.resize(data_row_size);
    thrust::transform(
        thrust::device, thrust::make_counting_iterator<tuple_size_t>(0),
        thrust::make_counting_iterator<tuple_size_t>(data_row_size),
        tuples_vec.begin(),
        [raw_ptr = this->data_raw, arity = this->arity] __device__(
            tuple_size_t i) -> tuple_type { return raw_ptr + i * arity; });
    tuples = tuples_vec.data().get();
    // TODO: use thrust tabulate to init tuples instead of transform
    this->tuple_counts = data_row_size;
}

void GHashRelContainer::build_index(int grid_size, int block_size) {
    // clear old index if exists
    if (this->index_map != nullptr) {
        checkCuda(cudaFree(this->index_map));
        this->index_map = nullptr;
    }
    // init the index map
    // set the size of index map same as data, (this should give us almost
    // no conflict) however this can be memory inefficient
    this->index_map_size =
        std::ceil(this->tuple_counts / this->index_map_load_factor);
    // this->index_map_size = data_row_size;
    u64 index_map_mem_size = this->index_map_size * sizeof(MEntity);
    checkCuda(cudaMalloc((void **)&(this->index_map), index_map_mem_size));
    checkCuda(cudaMemset(this->index_map, 0, index_map_mem_size));

    GHashRelContainer *target_device;
    checkCuda(cudaMalloc((void **)&target_device, sizeof(GHashRelContainer)));
    checkCuda(cudaMemcpy(target_device, this, sizeof(GHashRelContainer),
                         cudaMemcpyHostToDevice));
    // load inited data struct into GPU memory
    thrust::fill(thrust::device, this->index_map,
                 this->index_map + this->index_map_size,
                 MEntity{EMPTY_HASH_ENTRY, EMPTY_HASH_ENTRY});
    calculate_index_hash<<<grid_size, block_size>>>(
        target_device,
        tuple_indexed_less(this->index_column_size, this->arity));
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(target_device));
}

void GHashRelContainer::reconstruct() {}

void GHashRelContainer::fit() {
    if (this->tuple_counts == 0) {
        return;
    }
    if (this->tuple_counts == this->data_raw_row_size) {
        return;
    }
    column_type *new_data;
    checkCuda(cudaMalloc((void **)&new_data, this->arity * this->tuple_counts *
                                                 sizeof(column_type)));
    thrust::for_each(
        thrust::device, thrust::make_counting_iterator<tuple_size_t>(0),
        thrust::make_counting_iterator<tuple_size_t>(this->tuple_counts),
        [gh_tps = this->tuples, arity = this->arity,
         new_data] __device__(tuple_size_t i) {
            for (int j = 0; j < arity; j++) {
                new_data[i * arity + j] = gh_tps[i][j];
            }
        });
    reload(new_data, this->tuple_counts);
}

void Relation::defragement(RelationVersion ver, int grid_size, int block_size) {
    GHashRelContainer *gh;
    if (ver == FULL) {
        if (buffered_delta_vectors.size() == 0 &&
            full->tuple_counts == full->data_raw_row_size) {
            return;
        }
        gh = full;
    } else if (ver == DELTA) {
        gh = delta;
    } else if (ver == NEWT) {
        gh = newt;
    }
    if (gh->tuple_counts == 0) {
        return;
    }
    gh->fit();
    // print_tuple_rows(gh, "after defragment");
    if (ver == NEWT) {
        return;
    }
    if (index_flag) {
        full->build_index(grid_size, block_size);
    }
    if (ver == FULL) {
        if (buffered_delta_vectors.size() <= 1) {
            return;
        }
        for (int j = 0; j < buffered_delta_vectors.size() - 1; j++) {
            buffered_delta_vectors[j]->free();
        }
    }
}

bool is_number(const std::string &s) {
    for (char const &c : s) {
        if (std::isdigit(c) == 0) {
            return false;
        }
    }
    return true;
}

std::string trim(const std::string &str) {
    size_t first = str.find_first_not_of(" \t\n");
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(" \t\f\v\n\r");
    return str.substr(first, (last - first + 1));
}

void file_to_buffer(std::string file_path,
                    thrust::host_vector<column_type> &buffer,
                    std::map<column_type, std::string> &string_map) {
    std::ifstream file(file_path);
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        // if empty line, skip
        if (line.empty()) {
            continue;
        }
        while (std::getline(iss, token, '\t')) {
            // if token is number (including hex start with 0x)
            auto trimed_token = trim(token);
            if (is_number(trimed_token)) {
                buffer.push_back(std::stoull(token));
            } else if (token.find("0x") == 0) {
                buffer.push_back(std::stoull(token, 0, 16));
            } else {
                // check if empty string
                if (token.empty()) {
                    buffer.push_back(0);
                }
                // if token is a string in value of the map, use the key
                // else use the hash value of the string as the key inserted
                // into the map
                column_type token_hash = std::hash<std::string>{}(token);
                auto it = string_map.find(token_hash);
                if (it == string_map.end()) {
                    string_map[token_hash] = token;
                    buffer.push_back(token_hash);
                } else {
                    buffer.push_back(it->first);
                }
            }
        }
    }
}
