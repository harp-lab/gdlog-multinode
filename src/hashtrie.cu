
#include "../include/hashtrie.cuh"

#include <thrust/adjacent_difference.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

namespace hisa {

// hashtrie::load(const thrust::host_vector<internal_data_type> &tuple) {

// }

// write a gpu functor kernel
// merge join
// merge 2 sort key ranges, a, b (both are not unique)
// if the key b is in a, set the result value to b
// if b key not in a, set result value to 0

// the first device functor is to compute the size of the join
// takes in a and b, parallel add the atomic size counter

struct MergeJoinSize {
    internal_data_type *a;
    internal_data_type *b;
    size_t *size;

    MergeJoinSize(internal_data_type *a, internal_data_type *b, size_t *size)
        : a(a), b(b), size(size) {}

    __device__ void operator()(size_t i) {}
};

void hisa_cpu::load(const thrust::host_vector<tuple_type> &tuples) {

    this->columns.resize(arity);

    for (int i = 0; i < arity; i++) {
        // extract the i-th column
        thrust::host_vector<internal_data_type> column_data(tuples.size());
        thrust::transform(tuples.begin(), tuples.end(), column_data.begin(),
                          [i](tuple_type tuple) { return tuple[i]; });
        // save columns raw
        columns[i].raw_data = column_data;
    }
    indexed = false;
    this->total_tuples = tuples.size();
}

void hisa_cpu::load_vectical(
    thrust::host_vector<thrust::host_vector<internal_data_type>> &tuples) {

    this->columns.resize(arity);

    for (int i = 0; i < arity; i++) {
        // extract the i-th column
        thrust::host_vector<internal_data_type> column_data(tuples[i].size());
        thrust::copy(tuples[i].begin(), tuples[i].end(), column_data.begin());
        // save columns raw
        columns[i].raw_data.swap(column_data);
    }
    indexed = false;
    total_tuples = tuples[0].size();
}

void hisa_cpu::deduplicate() {
    // radix sort the raw data of each column
    thrust::host_vector<uint32_t> lexical_order_indices(total_tuples);
    thrust::sequence(lexical_order_indices.begin(),
                     lexical_order_indices.end());
    thrust::host_vector<internal_data_type> tmp_raw(total_tuples);
    for (int i = arity - 1; i >= 0; i--) {
        auto &column = columns[i];
        auto &column_data = column.raw_data;
        // gather the column data
        thrust::gather(lexical_order_indices.begin(),
                       lexical_order_indices.end(), column_data.begin(),
                       tmp_raw.begin());
        thrust::stable_sort_by_key(tmp_raw.begin(), tmp_raw.end(),
                                   lexical_order_indices.begin());
        if (i == 0) {
            column_data.swap(tmp_raw);
        }
    }
    for (int i = 1; i < arity; i++) {
        thrust::gather(lexical_order_indices.begin(),
                       lexical_order_indices.end(), columns[i].raw_data.begin(),
                       tmp_raw.begin());
        columns[i].raw_data.swap(tmp_raw);
    }
    // remove lexical_order_indices
    lexical_order_indices.resize(0);
    lexical_order_indices.shrink_to_fit();
    tmp_raw.resize(0);
    tmp_raw.shrink_to_fit();

    thrust::host_vector<bool> dup_flags(total_tuples, true);
    dup_flags[0] = false;
    for (int i = 0; i < arity; i++) {
        thrust::host_vector<bool> cur_col_dup_flags(total_tuples, false);
        thrust::adjacent_difference(
            columns[i].raw_data.begin(), columns[i].raw_data.end(),
            cur_col_dup_flags.begin(), thrust::equal_to<internal_data_type>());
        thrust::transform(cur_col_dup_flags.begin(), cur_col_dup_flags.end(),
                          dup_flags.begin(), dup_flags.begin(),
                          thrust::logical_and<bool>());
    }

    // remove dup in each column using gather
    uint32_t new_size = 0;
    for (int i = 0; i < arity; i++) {
        auto new_col_end = thrust::remove_if(
            columns[i].raw_data.begin(), columns[i].raw_data.end(),
            dup_flags.begin(), thrust::identity<bool>());
        new_size = new_col_end - columns[i].raw_data.begin();
        columns[i].raw_data.resize(new_size);
    }
    total_tuples = new_size;
}

void hisa_cpu::build_index(bool sorted) {
    if (indexed) {
        return;
    } else {
        for (int i = 0; i < arity; i++) {
            // std::cout << "build index " << i << std::endl;
            thrust::host_vector<internal_data_type> column_data(total_tuples);
            if (!sorted) {
                columns[i].sorted_indices.resize(total_tuples);
                thrust::sequence(columns[i].sorted_indices.begin(),
                                 columns[i].sorted_indices.end());
                // sort all values in the column
                thrust::copy(columns[i].raw_data.begin(),
                             columns[i].raw_data.end(), column_data.begin());
                thrust::stable_sort_by_key(column_data.begin(),
                                           column_data.end(),
                                           columns[i].sorted_indices.begin());
            } else {
                thrust::gather(columns[i].sorted_indices.begin(),
                               columns[i].sorted_indices.end(),
                               columns[i].raw_data.begin(),
                               column_data.begin());
            }
            // compress the column, save unique values and their counts
            thrust::host_vector<uint32_t> unique_offset(total_tuples);
            thrust::sequence(unique_offset.begin(), unique_offset.end());
            // using thrust parallel algorithm to compress the column
            // mark non-unique values as 0
            // print column_data
            // for (int j = 0; j < column_data.size(); j++) {
            //     std::cout << column_data[j] << " ";
            // }
            // std::cout << std::endl;

            auto uniq_end = thrust::unique_by_key(
                column_data.begin(), column_data.end(), unique_offset.begin());
            auto uniq_size = uniq_end.first - column_data.begin();
            // column_data.resize(uniq_size);
            // unique_offset.resize(uniq_size);
            columns[i].unique_v = column_data;
            columns[i].unique_v.resize(uniq_size);
            // column_data.resize(0);
            // column_data.shrink_to_fit();

            // update map
            auto &vmap = columns[i].unique_v_map;
            auto start = std::chrono::high_resolution_clock::now();
            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(
                    columns[i].unique_v.begin(), unique_offset.begin())),
                thrust::make_zip_iterator(
                    thrust::make_tuple(columns[i].unique_v.end(),
                                       unique_offset.begin() + uniq_size)),
                [&vmap](auto &t) {
                    auto &value = thrust::get<0>(t);
                    auto &offset = thrust::get<1>(t);
                    Map::accessor res;
                    // vmap[value] = offset;
                    vmap.insert(res, std::make_pair(value, offset));
                });
            // for (int j = 0; j < uniq_size; j++) {
            //     vmap[column_data[j]] = unique_offset[j];
            // }
            auto end = std::chrono::high_resolution_clock::now();
            this->hash_time +=
                std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                      start)
                    .count();

            unique_offset.resize(uniq_size);
            unique_offset.push_back(total_tuples);
            // calculate offset by minus the previous value
            thrust::adjacent_difference(unique_offset.begin(),
                                        unique_offset.end(),
                                        unique_offset.begin());
            // the first value is always 0, in following code, we will use the
            // 1th as the start index
            unique_offset.erase(unique_offset.begin());
            // thrust::host_vector<internal_data_type> next_indices(
            //     column_data.size());
            // thrust::exclusive_scan(unique_offset.begin(),
            // unique_offset.end(),
            //                        next_indices.begin());

            unique_offset.resize(uniq_size);
            columns[i].v_offset.swap(unique_offset);
        }
    }
    indexed = true;
}

void hisa_cpu::fetch_column_values(
    int column, thrust::host_vector<internal_data_type> &values, bool sorted) {
    // if (!indexed) {
    //     throw std::runtime_error("hashtrie is not indexed");
    // }
    if (column >= arity) {
        throw std::runtime_error("column index out of range");
    }
    auto &column_data = columns[column].raw_data;
    auto &indices = columns[column].sorted_indices;
    values.resize(column_data.size());
    if (sorted) {
        thrust::gather(indices.begin(), indices.end(), column_data.begin(),
                       values.begin());
    } else {
        thrust::copy(column_data.begin(), column_data.end(), values.begin());
    }
}

void hisa_cpu::fetch_column_unique_values(
    int column, thrust::host_vector<internal_data_type> &values) {
    if (!indexed) {
        throw std::runtime_error("hashtrie is not indexed");
    }
    if (column >= arity) {
        throw std::runtime_error("column index out of range");
    }
    values.resize(columns[column].unique_v.size());
    thrust::copy(columns[column].unique_v.begin(),
                 columns[column].unique_v.end(), values.begin());
}

void hisa_cpu::remove_dup_in(hisa_cpu &other) {

    thrust::host_vector<uint32_t> dup_indices(other.get_total_tuples());
    thrust::sequence(dup_indices.begin(), dup_indices.end());

    thrust::host_vector<internal_data_type> other_column_data(
        dup_indices.size());

    for (int i = 0; i < arity; i++) {
        auto &column = columns[i];
        auto &other_column = other.columns[i];

        // use permutation iterator?
        thrust::gather(dup_indices.begin(), dup_indices.end(),
                       other_column.raw_data.begin(),
                       other_column_data.begin());

        thrust::host_vector<bool> diff_flags(dup_indices.size(), false);
        // TODO: diff here
        auto &unique_v_map = column.unique_v_map;
        thrust::transform(other_column_data.begin(), other_column_data.end(),
                          diff_flags.begin(),
                          [&unique_v_map](internal_data_type value) -> bool {
                              Map::accessor res;
                              return unique_v_map.find(res, value);
                          });

        auto diff_end =
            thrust::remove_if(dup_indices.begin(), dup_indices.end(),
                              diff_flags.begin(), thrust::identity<bool>());

        auto size_after_diff = diff_end - dup_indices.begin();
        dup_indices.resize(size_after_diff);
        other_column_data.resize(size_after_diff);
    }

    thrust::host_vector<bool> indice_flags(other.get_total_tuples(), false);
    // set all permute indices in dup_indices to true
    thrust::transform(thrust::make_permutation_iterator(indice_flags.begin(),
                                                        dup_indices.begin()),
                      thrust::make_permutation_iterator(indice_flags.begin(),
                                                        dup_indices.end()),
                      indice_flags.begin(), [](auto &i) { return true; });

    // remove dup in other
    for (int i = 0; i < arity; i++) {
        auto &column = other.columns[i];
        auto &indices = column.sorted_indices;
        auto &raw_data = column.raw_data;
        auto &unique_v = column.unique_v;
        auto &v_offset = column.v_offset;
        auto &unique_v_map = column.unique_v_map;

        auto index_rm_it_begin = thrust::make_permutation_iterator(
            indice_flags.begin(), indices.begin());

        auto index_rm_it_end = thrust::make_permutation_iterator(
            indice_flags.begin(), indices.end());

        // remove at indices
        // set the things need to remove to MAX
        thrust::transform(
            thrust::make_zip_iterator(
                thrust::make_tuple(indices.begin(), index_rm_it_begin)),
            thrust::make_zip_iterator(
                thrust::make_tuple(indices.end(), index_rm_it_end)),
            indices.begin(), [](auto &t) {
                auto &index = thrust::get<0>(t);
                auto &rm = thrust::get<1>(t);
                return rm ? std::numeric_limits<internal_data_type>::max()
                          : index;
            });

        // compute raw value need remove
        // thrust::host_vector<internal_data_type> raw_value_need_rm_sorted(
        //     dup_indices.size());
        // thrust::gather(dup_indices.begin(), dup_indices.end(),
        // raw_data.begin(),
        //                raw_value_need_rm_sorted.begin());
        // thrust::sort(raw_value_need_rm_sorted.begin(),
        //              raw_value_need_rm_sorted.end());
        // thrust::host_vector<uint32_t> raw_value_need_rm_count(
        //     dup_indices.size());
        // thrust::sequence(raw_value_need_rm_count.begin(),
        //                  raw_value_need_rm_count.end());
        // thrust::unique_by_key(raw_value_need_rm_sorted.begin(),
        //                       raw_value_need_rm_sorted.end(),
        //                       raw_value_need_rm_count.begin());
        // raw_value_need_rm_count.push_back(dup_indices.size());
        // thrust::adjacent_difference(raw_value_need_rm_count.begin(),
        //                             raw_value_need_rm_count.end(),
        //                             raw_value_need_rm_count.begin());
        // raw_value_need_rm_count.erase(raw_value_need_rm_count.begin());

        // update map
        // thrust::for_each(
        //     thrust::make_zip_iterator(
        //         thrust::make_tuple(raw_value_need_rm_sorted.begin(),
        //                            raw_value_need_rm_count.begin())),
        //     thrust::make_zip_iterator(thrust::make_tuple(
        //         raw_value_need_rm_sorted.end(),
        //         raw_value_need_rm_count.end())),
        //     [&unique_v, &unique_v_map](auto &t) {
        //         auto &value = thrust::get<0>(t);
        //         auto &count = thrust::get<1>(t);
        //         Map::accessor found_res;
        //         if (unique_v_map.find(found_res, value)) {
        //             auto new_v = found_res->second - count;
        //             Map::accessor write_res;
        //             auto insert_new = unique_v_map.insert(write_res, new_v);
        //             if (!insert_new) {
        //                 throw std::runtime_error("race condition !");
        //             }
        //         }
        //         if (unique_v_map.find(value) != unique_v_map.end()) {
        //             auto offset = unique_v_map[value];
        //             unique_v_map[value] = offset - count;
        //         } else {
        //             unique_v_map[value] = count;
        //         }
        //     });

        // remove in data_raw & sorted_indices
        auto new_indices_end =
            thrust::remove(indices.begin(), indices.end(),
                           std::numeric_limits<internal_data_type>::max());
        indices.resize(new_indices_end - indices.begin());

        auto new_end =
            thrust::remove_if(raw_data.begin(), raw_data.end(),
                              indice_flags.begin(), thrust::identity<bool>());
        auto new_size = new_end - raw_data.begin();
        raw_data.resize(new_size);
    }
    other.total_tuples = dup_indices.size();
    other.build_index(false);
}

void hisa_cpu::merge(hisa_cpu &other) {
    if (!indexed) {
        throw std::runtime_error("hashtrie is not indexed");
    }

    for (int i = 0; i < arity; i++) {
        auto &column = columns[i];
        auto &other_column = other.columns[i];

        auto old_len = column.raw_data.size();
        column.raw_data.insert(column.raw_data.end(),
                               other_column.raw_data.begin(),
                               other_column.raw_data.end());
        // column.raw_data.resize(old_len + other_column.raw_data.size());
        // thrust::gather(other_column.sorted_indices.begin(),
        //                other_column.sorted_indices.end(),
        //                other_column.raw_data.begin(),
        //                column.raw_data.begin() + old_len);

        // move out raw_data from other_column
        other_column.raw_data.resize(0);
        other_column.raw_data.shrink_to_fit();

        thrust::host_vector<uint32_t> merged_indices(column.raw_data.size());
        // add size of column_raw to the indices in other_column
        thrust::transform(other_column.sorted_indices.begin(),
                          other_column.sorted_indices.end(),
                          thrust::make_constant_iterator(total_tuples),
                          other_column.sorted_indices.begin(),
                          thrust::plus<internal_data_type>());

        // print merged_indices

        thrust::merge_by_key(
            thrust::make_permutation_iterator(column.raw_data.begin(),
                                              column.sorted_indices.begin()),
            thrust::make_permutation_iterator(column.raw_data.begin(),
                                              column.sorted_indices.begin() +
                                                  old_len),
            thrust::make_permutation_iterator(
                column.raw_data.begin(), other_column.sorted_indices.begin()),
            thrust::make_permutation_iterator(
                column.raw_data.begin(), other_column.sorted_indices.end()),
            column.sorted_indices.begin(), other_column.sorted_indices.begin(),
            thrust::make_discard_iterator(), merged_indices.begin());
        column.sorted_indices.swap(merged_indices);

        // drop other_column
        other_column.sorted_indices.resize(0);
        other_column.sorted_indices.shrink_to_fit();
        other_column.unique_v.resize(0);
        other_column.unique_v.shrink_to_fit();
        other_column.v_offset.resize(0);
        other_column.v_offset.shrink_to_fit();
        other_column.unique_v_map.clear();
    }
    total_tuples += other.total_tuples;

    // TODO: optimize this
    build_index(true);
}

void hisa_cpu::print_all(bool sorted) {
    // print all columns
    thrust::host_vector<internal_data_type> column(total_tuples);
    for (int i = 0; i < arity; i++) {
        fetch_column_values(i, column, sorted);
        std::cout << "column data " << i << " "
                  << columns[i].sorted_indices.size() << ":\n";
        for (int j = 0; j < column.size(); j++) {
            std::cout << column[j] << " ";
        }
        std::cout << std::endl;
        std::cout << "unique values " << i << " " << columns[i].unique_v.size()
                  << ":\n";
        for (int j = 0; j < columns[i].unique_v.size(); j++) {
            std::cout << columns[i].unique_v[j] << " ";
        }
        std::cout << std::endl;
        // std::cout << "unique offset " << i << " " <<
        // columns[i].v_offset.size() << ":\n"; for (int j = 0; j <
        // columns[i].v_offset.size(); j++) {
        //     std::cout << columns[i].v_offset[j] << " ";
        // }
        // std::cout << std::endl;
        for (auto &pair : columns[i].unique_v_map) {
            std::cout << "(" << pair.first << " " << pair.second << ") ";
        }
        std::cout << std::endl;
    }
}

void hisa_cpu::clear() {
    for (int i = 0; i < arity; i++) {
        columns[i].raw_data.resize(0);
        columns[i].raw_data.shrink_to_fit();
        columns[i].sorted_indices.resize(0);
        columns[i].sorted_indices.shrink_to_fit();
        columns[i].unique_v.resize(0);
        columns[i].unique_v.shrink_to_fit();
        columns[i].v_offset.resize(0);
        columns[i].v_offset.shrink_to_fit();
        columns[i].unique_v_map.clear();
    }
    total_tuples = 0;
    indexed = false;
}

// simple map
void GpuSimplMap::insert(device_data_t &keys, device_ranges_t &values) {
    // swap in
    this->keys.swap(keys);
    this->values.swap(values);
}

void GpuSimplMap::find(device_data_t &keys, device_ranges_t &result) {
    // keys is the input, values is the output
    result.resize(keys.size());
    // device_data_t found_keys(keys.size());

    thrust::transform(
        keys.begin(), keys.end(), result.begin(),
        [map_keys = this->keys.data().get(), map_vs = this->values.data().get(),
         ksize = this->keys.size()] __device__(internal_data_type key)
            -> comp_range_t {
            auto it = thrust::lower_bound(thrust::seq, map_keys,
                                          map_keys + ksize, key);
            return map_vs[it - map_keys];
        });
}

// multi_hisa
void VerticalColumnGpu::clear_unique_v() {
    if (!unique_v_map) {
        // delete unique_v_map;
        unique_v_map = nullptr;
    }
}

VerticalColumnGpu::~VerticalColumnGpu() { clear_unique_v(); }

multi_hisa::multi_hisa(int arity) {
    this->arity = arity;
    newt_size = 0;
    full_size = 0;
    delta_size = 0;
    full_columns.resize(arity);
    delta_columns.resize(arity);
    newt_columns.resize(arity);
    data.resize(arity);
}

void multi_hisa::init_load_vectical(
    thrust::host_vector<thrust::host_vector<internal_data_type>> &tuples,
    size_t rows) {
    auto load_start = std::chrono::high_resolution_clock::now();
    auto total_tuples = tuples[0].size();
    for (int i = 0; i < arity; i++) {
        // extract the i-th column
        // thrust::device_vector<internal_data_type> column_data(total_tuples);
        data[i].resize(total_tuples);
        cudaMemcpy(data[i].data().get(), tuples[i].data(),
                   tuples[i].size() * sizeof(internal_data_type),
                   cudaMemcpyHostToDevice);
        // save columns raw
    }
    this->total_tuples = total_tuples;
    this->newt_size = total_tuples;
    // set newt
    for (int i = 0; i < arity; i++) {
        newt_columns[i].raw_data = data[i].data();
    }
    auto load_end = std::chrono::high_resolution_clock::now();
    this->load_time += std::chrono::duration_cast<std::chrono::microseconds>(
                           load_end - load_start)
                           .count();
}

void multi_hisa::print_all(bool sorted) {
    // print all columns in full
    thrust::host_vector<internal_data_type> column(total_tuples);
    thrust::host_vector<internal_data_type> unique_value(total_tuples);
    for (int i = 0; i < arity; i++) {
        thrust::copy(full_columns[i].raw_data,
                     full_columns[i].raw_data + full_size, column.begin());
        std::cout << "column data " << i << " " << total_tuples << ":\n";
        for (int j = 0; j < column.size(); j++) {
            std::cout << column[j] << " ";
        }
        std::cout << std::endl;
        std::cout << "unique values " << i << " "
                  << full_columns[i].unique_v.size() << ":\n";
        unique_value.resize(full_columns[i].unique_v.size());
        thrust::copy(full_columns[i].unique_v.begin(),
                     full_columns[i].unique_v.end(), unique_value.begin());
        for (int j = 0; j < full_columns[i].unique_v.size(); j++) {
            std::cout << unique_value[j] << " ";
        }
        std::cout << std::endl;
    }
}

void multi_hisa::build_index(RelationVersion version, bool sorted) {
    auto &columns = get_versioned_columns(version);
    auto version_size = get_versioned_size(version);

    device_data_t unique_offset(total_tuples);
    device_data_t unique_diff(total_tuples);
    for (size_t i = 0; i < arity; i++) {
        device_data_t column_data(total_tuples);
        // std::cout << "build index " << i << std::endl;
        auto sorte_start = std::chrono::high_resolution_clock::now();
        if (!sorted) {
            columns[i].sorted_indices.resize(total_tuples);
            thrust::sequence(DEFAULT_DEVICE_POLICY,
                             columns[i].sorted_indices.begin(),
                             columns[i].sorted_indices.end());
            // sort all values in the column
            thrust::copy(DEFAULT_DEVICE_POLICY, columns[i].raw_data,
                         columns[i].raw_data + version_size,
                         column_data.begin());
            // if (i != 0) {
            thrust::sort_by_key(DEFAULT_DEVICE_POLICY, column_data.begin(),
                                column_data.end(),
                                columns[i].sorted_indices.begin());
            // }
        } else {
            thrust::gather(DEFAULT_DEVICE_POLICY,
                           columns[i].sorted_indices.begin(),
                           columns[i].sorted_indices.end(), columns[i].raw_data,
                           column_data.begin());
        }
        auto sort_end = std::chrono::high_resolution_clock::now();
        this->sort_time +=
            std::chrono::duration_cast<std::chrono::microseconds>(sort_end -
                                                                  sorte_start)
                .count();
        // compress the column, save unique values and their counts
        thrust::sequence(DEFAULT_DEVICE_POLICY, unique_offset.begin(),
                         unique_offset.end());
        // using thrust parallel algorithm to compress the column
        // mark non-unique values as 0
        // print column_data
        // for (int j = 0; j < column_data.size(); j++) {
        //     std::cout << column_data[j] << " ";
        // }s
        // std::cout << std::endl;

        auto uniq_end =
            thrust::unique_by_key(DEFAULT_DEVICE_POLICY, column_data.begin(),
                                  column_data.end(), unique_offset.begin());
        auto uniq_size = uniq_end.first - column_data.begin();
        // column_data.resize(uniq_size);
        // unique_offset.resize(uniq_size);
        // columns[i].unique_v.swap(column_data);
        auto &uniq_val = column_data;
        uniq_val.resize(uniq_size);
        // columns[i].unique_v.resize(uniq_size);
        // thrust::copy(DEFAULT_DEVICE_POLICY, column_data.begin(),
        //              column_data.begin() + uniq_size,
        //              columns[i].unique_v.begin());
        // column_data.resize(0);
        // column_data.shrink_to_fit();

        unique_offset.resize(uniq_size);
        // device_data_t unique_diff = unique_offset;
        thrust::copy(DEFAULT_DEVICE_POLICY, unique_offset.begin(),
                     unique_offset.end(), unique_diff.begin());
        unique_diff.resize(uniq_size);
        unique_diff.push_back(total_tuples);
        // calculate offset by minus the previous value
        thrust::adjacent_difference(DEFAULT_DEVICE_POLICY, unique_diff.begin(),
                                    unique_diff.end(), unique_diff.begin());
        // the first value is always 0, in following code, we will use the
        // 1th as the start index
        unique_diff.erase(unique_diff.begin());
        unique_diff.resize(uniq_size);

        // update map
        auto start = std::chrono::high_resolution_clock::now();
        if (columns[i].use_real_map) {
            if (columns[i].unique_v_map) {
                // columns[i].unique_v_map->reserve(uniq_size);
                columns[i].unique_v_map = nullptr;
                columns[i].unique_v_map = CREATE_V_MAP(uniq_size);
            } else {
                columns[i].unique_v_map = CREATE_V_MAP(uniq_size);
            }
            auto insertpair_begin = thrust::make_transform_iterator(
                thrust::make_zip_iterator(
                    thrust::make_tuple(uniq_val.begin(), unique_offset.begin(),
                                       unique_diff.begin())),
                cuda::proclaim_return_type<GpuMapPair>([] __device__(auto &t) {
                    return HASH_NAMESPACE::make_pair(
                        thrust::get<0>(t),
                        (static_cast<uint64_t>(thrust::get<1>(t)) << 32) +
                            (static_cast<uint64_t>(thrust::get<2>(t))));
                }));
            columns[i].unique_v_map->insert(insertpair_begin,
                                            insertpair_begin + uniq_size);
        } else {
            device_ranges_t ranges(uniq_size);
            // compress offset and diff to u64 ranges
            thrust::transform(
                DEFAULT_DEVICE_POLICY, unique_offset.begin(),
                unique_offset.end(), unique_diff.begin(), ranges.begin(),
                [] __device__(auto &offset, auto &diff) -> uint64_t {
                    return (static_cast<uint64_t>(offset) << 32) +
                           (static_cast<uint64_t>(diff));
                });
            columns[i].unique_v_map_simp.insert(uniq_val, ranges);
            // std::cout << "ranges size: " << ranges.size() << std::endl;
        }
        auto end = std::chrono::high_resolution_clock::now();
        this->hash_time +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();
    }
    indexed = true;
}

void multi_hisa::deduplicate() {
    auto start = std::chrono::high_resolution_clock::now();
    VersionedColumns &columns = newt_columns;
    auto version_size = newt_size;
    // radix sort the raw data of each column
    device_data_t dup_indices(version_size);
    thrust::sequence(DEFAULT_DEVICE_POLICY, dup_indices.begin(),
                     dup_indices.end());
    device_data_t tmp_raw(version_size);
    thrust::device_vector<bool> dup_flags(version_size, false);
    for (int i = arity - 1; i >= 0; i--) {
        auto &column = columns[i];
        auto &column_data = column.raw_data;
        // gather the column data
        thrust::gather(DEFAULT_DEVICE_POLICY, dup_indices.begin(),
                       dup_indices.end(), column_data, tmp_raw.begin());
        thrust::stable_sort_by_key(DEFAULT_DEVICE_POLICY, tmp_raw.begin(),
                                   tmp_raw.end(), dup_indices.begin());
        // check if a value is duplicated to the left & right
        if (i != 0) {
            thrust::transform(
                DEFAULT_DEVICE_POLICY,
                thrust::make_counting_iterator<uint32_t>(0),
                thrust::make_counting_iterator<uint32_t>(dup_flags.size()),
                dup_flags.begin(),
                [tmp_raw = tmp_raw.data().get(),
                 total = dup_flags.size()] __device__(uint32_t i) -> bool {
                    if (i == 0) {
                        return tmp_raw[i] == tmp_raw[i + 1];
                    } else if (i == total - 1) {
                        return tmp_raw[i] == tmp_raw[i - 1];
                    } else {
                        return (tmp_raw[i] == tmp_raw[i - 1]) ||
                               (tmp_raw[i] == tmp_raw[i + 1]);
                    }
                });
        } else {
            // last column, we only need check the left, because the right is
            // what we want to keep
            thrust::transform(
                DEFAULT_DEVICE_POLICY,
                thrust::make_counting_iterator<uint32_t>(0),
                thrust::make_counting_iterator<uint32_t>(dup_flags.size()),
                dup_flags.begin(),
                [tmp_raw = tmp_raw.data().get(),
                 total = dup_flags.size()] __device__(uint32_t i) -> bool {
                    if (i == 0) {
                        return false;
                    } else {
                        return tmp_raw[i] == tmp_raw[i - 1];
                    }
                });
        }

        // print tmp_raw here
        // std::cout << "tmp_raw:\n";
        // thrust::host_vector<internal_data_type> h_tmp_raw = tmp_raw;
        // for (int i = 0; i < h_tmp_raw.size(); i++) {
        //     std::cout << h_tmp_raw[i] << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "dup_indices:\n";
        // thrust::host_vector<internal_data_type> h_dup_indices = dup_indices;
        // for (int j = 0; j < h_dup_indices.size(); j++) {
        //     std::cout << h_dup_indices[j] << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "dup_flags:\n";
        // thrust::host_vector<bool> h_dup_flags = dup_flags;
        // for (int j = 0; j < h_dup_flags.size(); j++) {
        //     std::cout << h_dup_flags[j] << " ";
        // }
        // std::cout << std::endl;
        //  filter keep only duplicates, the one whose dup_flags is true
        auto new_dup_indices_end = thrust::remove_if(
            DEFAULT_DEVICE_POLICY, dup_indices.begin(), dup_indices.end(),
            dup_flags.begin(), thrust::logical_not<bool>());
        auto new_dup_size = new_dup_indices_end - dup_indices.begin();
        dup_indices.resize(new_dup_size);
        // resize and reset flags
        dup_flags.resize(new_dup_size);
        tmp_raw.resize(new_dup_size);
        if (new_dup_size == 0) {
            break;
        }
    }
    // print dup_indices
    // thrust::host_vector<internal_data_type> h_dup_indices = dup_indices;
    // for (int i = 0; i < h_dup_indices.size(); i++) {
    //     std::cout << h_dup_indices[i] << " ";
    // }
    // std::cout << std::endl;

    // tmp_raw.resize(0);
    // tmp_raw.shrink_to_fit();

    for (int i = 0; i < arity; i++) {
        thrust::fill(DEFAULT_DEVICE_POLICY,
                     thrust::make_permutation_iterator(columns[i].raw_data,
                                                       dup_indices.begin()),
                     thrust::make_permutation_iterator(columns[i].raw_data,
                                                       dup_indices.end()),
                     UINT32_MAX);
    }

    // remove dup in raw_data
    uint32_t new_size = 0;
    for (int i = 0; i < arity; i++) {
        auto new_col_end =
            thrust::remove(DEFAULT_DEVICE_POLICY, columns[i].raw_data,
                           columns[i].raw_data + version_size, UINT32_MAX);
        new_size = new_col_end - columns[i].raw_data;
        columns[i].indexed = false;
    }
    newt_size = new_size;
    total_tuples = newt_size + full_size;
    for (size_t i = 0; i < arity; i++) {
        data[i].resize(total_tuples);
    }
    indexed = false;
    auto end = std::chrono::high_resolution_clock::now();
    dedup_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
}

void multi_hisa::fit() {
    total_tuples = newt_size + full_size;
    for (int i = 0; i < arity; i++) {
        data[i].resize(total_tuples);
        data[i].shrink_to_fit();
    }
}

void multi_hisa::new_to_delta() {
    // if (delta_size != 0) {
    //     throw std::runtime_error("delta is not empty before load new to
    //     delta");
    // }

    delta_size = newt_size;
    thrust::swap(newt_columns, delta_columns);
    // This is redundant
    for (int i = 0; i < arity; i++) {
        newt_columns[i].raw_data = nullptr;
    }
}

void column_match(VerticalColumnGpu &column_inner,
                  VerticalColumnGpu &column_outer,
                  device_pairs_t &matched_pair) {
    auto matched_size = matched_pair.size();
    // device_data_t matched_newt_pos(matched_size);
    // thrust::transform(
    //     DEFAULT_DEVICE_POLICY, matched_pair.begin(), matched_pair.end(),
    //     matched_newt_pos.begin(),
    //     [] __device__(auto &t) { return static_cast<uint32_t>(t >> 32); });
    // device_data_t matched_full_pos(matched_size);
    // thrust::transform(
    //     DEFAULT_DEVICE_POLICY, matched_pair.begin(), matched_pair.end(),
    //     matched_full_pos.begin(),
    //     [] __device__(auto &t) { return static_cast<uint32_t>(t); });
    // check if they agree on the value
    thrust::device_vector<bool> unmatched_flags(matched_size, true);
    // thrust::transform(
    //     thrust::make_permutation_iterator(column_b.raw_data,
    //                                       matched_newt_pos.begin()),
    //     thrust::make_permutation_iterator(column_b.raw_data,
    //                                       matched_newt_pos.end()),
    //     thrust::make_permutation_iterator(column_a.raw_data,
    //                                       matched_full_pos.begin()),
    //     unmatched_flags.begin(), thrust::not_equal_to<internal_data_type>());
    device_data_t tmp_inner(matched_size);
    thrust::transform(DEFAULT_DEVICE_POLICY, matched_pair.begin(),
                      matched_pair.end(), tmp_inner.begin(),
                      [raw_inner = column_inner.raw_data.get()] __device__(
                          auto &t) { return raw_inner[t & 0xFFFFFFFF]; });
    // print tmp_inner
    thrust::host_vector<internal_data_type> h_tmp_inner = tmp_inner;
    for (int i = 0; i < h_tmp_inner.size(); i++) {
        std::cout << h_tmp_inner[i] << " ";
    }
    std::cout << std::endl;

    thrust::transform(
        matched_pair.begin(), matched_pair.end(), unmatched_flags.begin(),
        [raw_outer = column_outer.raw_data.get(),
         raw_inner = column_inner.raw_data.get()] __device__(auto &t) {
            return raw_outer[t >> 32] != raw_inner[t & 0xFFFFFFFF];
        });
    // filter
    auto new_matched_pair_end =
        thrust::remove_if(matched_pair.begin(), matched_pair.end(),
                          unmatched_flags.begin(), thrust::identity<bool>());
    auto new_matched_pair_size = new_matched_pair_end - matched_pair.begin();
    matched_pair.resize(new_matched_pair_size);
}

void multi_hisa::persist_newt() {
    // clear the index of delta
    for (int i = 0; i < arity; i++) {
        delta_columns[i].sorted_indices.resize(0);
        delta_columns[i].sorted_indices.shrink_to_fit();
        delta_columns[i].unique_v.resize(0);
        delta_columns[i].unique_v.shrink_to_fit();
        delta_columns[i].clear_unique_v();
        delta_columns[i].unique_v_map = nullptr;
        delta_columns[i].raw_data = nullptr;
    }
    // merge newt to full
    if (full_size == 0) {
        full_size = newt_size;
        // thrust::swap(newt_columns, full_columns);
        for (int i = 0; i < arity; i++) {
            full_columns[i].raw_data = newt_columns[i].raw_data;
            newt_columns[i].raw_data = nullptr;
        }
        build_index(RelationVersion::FULL);
        return;
    }

    // difference newt and full, this a join
    // thrust::device_vector<internal_data_type> newt_full_diff(newt_size);
    device_pairs_t matched_pair;
    thrust::sequence(DEFAULT_DEVICE_POLICY, matched_pair.begin(),
                     matched_pair.end());
    device_data_t match_newt(newt_size);
    thrust::sequence(DEFAULT_DEVICE_POLICY, match_newt.begin(),
                     match_newt.end());
    for (size_t i = 0; i < arity; i++) {
        auto &newt_column = newt_columns[i];
        auto &full_column = full_columns[i];
        if (matched_pair.size() != 0) {
            column_match(full_column, newt_column, matched_pair);
            match_newt.resize(matched_pair.size());
            match_newt.shrink_to_fit();
            // populate the tuple need match later
            thrust::transform(
                DEFAULT_DEVICE_POLICY, matched_pair.begin(), matched_pair.end(),
                match_newt.begin(),
                cuda::proclaim_return_type<uint32_t>([] __device__(auto &t) {
                    return static_cast<uint32_t>(t >> 32);
                }));
        }
        if (match_newt.size() == 0) {
            break;
        }
        column_join(full_column, newt_column, match_newt, matched_pair);
    }

    // clear newt only keep match_newt
    if (match_newt.size() != 0) {
        device_data_t dup_newt_flags(newt_size, false);
        thrust::fill(DEFAULT_DEVICE_POLICY,
                     thrust::make_permutation_iterator(dup_newt_flags.begin(),
                                                       match_newt.begin()),
                     thrust::make_permutation_iterator(dup_newt_flags.begin(),
                                                       match_newt.end()),
                     true);
        auto new_newt_end =
            thrust::remove_if(DEFAULT_DEVICE_POLICY, newt_columns[0].raw_data,
                              newt_columns[0].raw_data + newt_size,
                              dup_newt_flags.begin(), thrust::identity<bool>());
        newt_size = new_newt_end - newt_columns[0].raw_data;
    }

    // sort and merge newt
    device_data_t merged_column(full_size + newt_size);
    device_data_t tmp_newt_v(newt_size);
    for (size_t i = 0; i < arity; i++) {
        auto &newt_column = newt_columns[i];
        auto &full_column = full_columns[i];

        newt_column.sorted_indices.resize(newt_size);
        thrust::copy(DEFAULT_DEVICE_POLICY, newt_column.raw_data,
                     newt_column.raw_data + newt_size, tmp_newt_v.begin());
        thrust::sequence(DEFAULT_DEVICE_POLICY,
                         newt_column.sorted_indices.begin(),
                         newt_column.sorted_indices.end(), full_size);
        thrust::sort_by_key(tmp_newt_v.begin(), tmp_newt_v.end(),
                            newt_column.sorted_indices.begin());

        // merge
        thrust::merge_by_key(
            DEFAULT_DEVICE_POLICY,
            thrust::make_permutation_iterator(
                full_column.raw_data, newt_column.sorted_indices.begin()),
            thrust::make_permutation_iterator(full_column.raw_data,
                                              newt_column.sorted_indices.end()),
            thrust::make_permutation_iterator(
                full_column.raw_data, full_column.sorted_indices.begin()),
            thrust::make_permutation_iterator(full_column.raw_data,
                                              full_column.sorted_indices.end()),
            newt_column.sorted_indices.begin(),
            full_column.sorted_indices.begin(), thrust::make_discard_iterator(),
            merged_column.begin());
        full_column.sorted_indices.swap(newt_column.sorted_indices);
        // minus size of full on all newt indices
        thrust::transform(DEFAULT_DEVICE_POLICY,
                          newt_column.sorted_indices.begin(),
                          newt_column.sorted_indices.end(),
                          thrust::make_constant_iterator(full_size),
                          newt_column.sorted_indices.begin(),
                          thrust::minus<internal_data_type>());
    }
    full_size += newt_size;

    // swap newt and delta
    delta_size = newt_size;
    newt_size = 0;
    // thrust::swap(newt_columns, delta_columns);
    for (int i = 0; i < arity; i++) {
        delta_columns[i].raw_data = newt_columns[i].raw_data;
        newt_columns[i].raw_data = nullptr;
    }

    // build indices on full
    build_index(RelationVersion::FULL, true);
    build_index(RelationVersion::DELTA, true);
}

void multi_hisa::clear() {
    for (int i = 0; i < arity; i++) {
        full_columns[i].raw_data = nullptr;
        full_columns[i].sorted_indices.resize(0);
        full_columns[i].sorted_indices.shrink_to_fit();
        full_columns[i].clear_unique_v();

        delta_columns[i].raw_data = nullptr;
        delta_columns[i].sorted_indices.resize(0);
        delta_columns[i].sorted_indices.shrink_to_fit();
        delta_columns[i].clear_unique_v();

        newt_columns[i].raw_data = nullptr;
        newt_columns[i].sorted_indices.resize(0);
        newt_columns[i].sorted_indices.shrink_to_fit();
        newt_columns[i].clear_unique_v();

        data[i].resize(0);
        data[i].shrink_to_fit();
    }
    total_tuples = 0;
}

void column_join(VerticalColumnGpu &inner_column,
                 VerticalColumnGpu &outer_column,
                 device_data_t &outer_tuple_indices,
                 device_pairs_t &matched_indices) {
    auto outer_size = outer_tuple_indices.size();

    device_ranges_t range_result(outer_tuple_indices.size());

    inner_column.unique_v_map->find(
        thrust::make_permutation_iterator(outer_column.raw_data,
                                          outer_tuple_indices.begin()),
        thrust::make_permutation_iterator(outer_column.raw_data,
                                          outer_tuple_indices.end()),
        range_result.begin());

    // print range_result
    // std::cout << "range_result:\n";
    // thrust::host_vector<uint64_t> h_range_result = range_result;
    // for (int i = 0; i < h_range_result.size(); i++) {
    //     std::cout << "(" << (h_range_result[i] >> 32) << " "
    //               << (h_range_result[i] & 0xffffffff) << ") ";
    // }
    // std::cout << std::endl;

    // materialize the comp_ranges

    // fecth all range size
    device_data_t size_vec(range_result.size());
    thrust::transform(
        DEFAULT_DEVICE_POLICY, range_result.begin(), range_result.end(),
        size_vec.begin(),
        [] __device__(auto &t) { return static_cast<uint32_t>(t); });

    device_ranges_t offset_vec;
    offset_vec.swap(range_result);
    thrust::transform(
        DEFAULT_DEVICE_POLICY, offset_vec.begin(), offset_vec.end(),
        offset_vec.begin(),
        [] __device__(auto &t) { return static_cast<uint64_t>(t >> 32); });

    uint32_t total_matched_size =
        thrust::reduce(DEFAULT_DEVICE_POLICY, size_vec.begin(), size_vec.end());
    device_data_t size_offset_tmp(outer_size);
    thrust::exclusive_scan(DEFAULT_DEVICE_POLICY, size_vec.begin(),
                           size_vec.end(), size_offset_tmp.begin());
    // std::cout << "total_matched_size: " << total_matched_size << std::endl;
    // // print size_vec
    // std::cout << "size_offset_tmp:\n";
    // thrust::host_vector<uint32_t> h_size_vec = offset_vec;
    // for (int i = 0; i < h_size_vec.size(); i++) {
    //     std::cout << h_size_vec[i] << " ";
    // }
    // std::cout << std::endl;

    // pirnt outer tuple indices
    // std::cout << "outer_tuple_indices:\n";
    // thrust::host_vector<uint32_t> h_outer_tuple_indices =
    // outer_tuple_indices; for (int i = 0; i < h_outer_tuple_indices.size();
    // i++) {
    //     std::cout << h_outer_tuple_indices[i] << " ";
    // }
    // std::cout << std::endl;

    // materialize the matched_indices
    matched_indices.resize(total_matched_size);
    thrust::for_each(
        DEFAULT_DEVICE_POLICY,
        thrust::make_zip_iterator(
            thrust::make_tuple(outer_tuple_indices.begin(), offset_vec.begin(),
                               size_vec.begin(), size_offset_tmp.begin())),
        thrust::make_zip_iterator(
            thrust::make_tuple(outer_tuple_indices.end(), offset_vec.end(),
                               size_vec.end(), size_offset_tmp.end())),
        [res = matched_indices.data().get(),
         inner_sorted_idx = inner_column.sorted_indices.data()
                                .get()] __device__(auto &t) -> comp_pair_t {
            auto outer_pos = thrust::get<0>(t);
            auto &inner_pos = thrust::get<1>(t);
            auto &size = thrust::get<2>(t);
            auto &start = thrust::get<3>(t);
            for (int i = 0; i < size; i++) {
                res[start + i] =
                    compress_u32(outer_pos, inner_sorted_idx[inner_pos + i]);
            }
        });
}

} // namespace hisa
