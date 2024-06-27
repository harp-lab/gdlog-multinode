
#include "../include/chisa.h"

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

} // namespace hisa
