#pragma once
#include "tuple.cuh"
#include "builtin.h"

#include <map>
#include <string>
#include <vector>
#include <thrust/host_vector.h>

#ifndef RADIX_SORT_THRESHOLD
#define RADIX_SORT_THRESHOLD 0
#endif
#ifndef FULL_BUFFER_VEC_MULTIPLIER
#define FULL_BUFFER_VEC_MULTIPLIER 5
#endif

enum RelationVersion { DELTA, FULL, NEWT };

/**
 * @brief A hash table entry
 * TODO: no need for struct actually, a u64[2] should be enough, easier to init
 *
 */
struct MEntity {
    // index position in actual index_arrary
    u64 key;
    // tuple position in actual data_arrary
    u64 value;
};

/**
 * @brief a C-style hashset indexing based relation container.
 *        Actual data is still stored using sorted set.
 *        Different from normal btree relation, using hash table storing the
 * index to accelarte range fetch. Good:
 *           - fast range fetch, in Shovon's ATC paper it shows great
 * performance.
 *           - fast serialization, its very GPU friendly and also easier for MPI
 * inter-rank comm transmission. Bad:
 *           - need reconstruct index very time tuple is inserted (need more
 * reasonable algorithm).
 *           - sorting is a issue, each update need resort everything seems
 * stupid.
 *
 */
struct GHashRelContainer {
    // open addressing hashmap for indexing
    MEntity *index_map = nullptr;
    tuple_size_t index_map_size = 0;
    float index_map_load_factor;

    // index prefix length
    // don't have to be u64,int is enough
    // u64 *index_columns;
    tuple_size_t index_column_size;

    // dependent postfix column always at the end of tuple
    int dependent_column_size = 0;

    // the pointer to flatten tuple, all tuple pointer here need to be sorted
    tuple_type *tuples = nullptr;
    // flatten tuple data
    column_type *data_raw = nullptr;
    // number of tuples
    tuple_size_t tuple_counts = 0;
    // actual tuple rows in flatten data, this maybe different from
    // tuple_counts when deduplicated
    tuple_size_t data_raw_row_size = 0;
    int arity;
    bool tmp_flag = false;

    GHashRelContainer(int arity, int indexed_column_size,
                      int dependent_column_size, bool tmp_flag = false)
        : arity(arity), index_column_size(indexed_column_size),
          dependent_column_size(dependent_column_size), tmp_flag(tmp_flag){};

    // TODO: impl this
    void reconstruct();

    // sort the tuple entries in the container
    void sort();

    // remove duplicate tuples
    void dedup();

    // reload data into the container (this won't sort/dedup and rebuild index)
    void reload(column_type *data, tuple_size_t data_row_size);

    // TODO: impl this, move construct hash table logic into this function
    void build_index(int grid_size, int block_size);

    // fit data_raw with tuple
    void fit();

    // clean everything in the container, drop all GPU memory
    void free();
};

enum JoinDirection { LEFT, RIGHT };

/**
 * @brief fill in index hash table for a relation in parallel, assume index is
 * correctly initialized, data has been loaded , deduplicated and sorted
 *
 * @param target the hashtable to init
 * @return dedeuplicated_bitmap
 */
__global__ void calculate_index_hash(GHashRelContainer *target,
                                     tuple_indexed_less cmp);

__global__ void get_copy_result(tuple_type *src_tuples,
                                column_type *dest_raw_data, int output_arity,
                                tuple_size_t tuple_counts,
                                TupleProjector tp_gen);

//////////////////////////////////////////////////////
// CPU functions

/**
 * @brief load raw data into relation container
 *
 * @param target hashtable struct in host
 * @param arity
 * @param data raw data on host
 * @param data_row_size
 * @param index_columns index columns id in host
 * @param index_column_size
 * @param index_map_load_factor
 * @param grid_size
 * @param block_size
 * @param gpu_data_flag if data is a GPU memory address directly assign to
 * target's data_raw
 * @param sorted_flag whether input raw data tuples are sorted (use sorted array
 * will be fasted, avoid extra sorting)
 * @param build_index_flag whether this relation container need indexing.
 */
void load_relation_container(
    GHashRelContainer *target, int arity, column_type *data,
    tuple_size_t data_row_size, tuple_size_t index_column_size,
    int dependent_column_size, float index_map_load_factor, int grid_size,
    int block_size, float *detail_time, bool gpu_data_flag = false,
    bool sorted_flag = false, bool build_index_flag = true,
    bool tuples_array_flag = true);

void repartition_relation_index(GHashRelContainer *target, int arity,
                                column_type *data, tuple_size_t data_row_size,
                                tuple_size_t index_column_size,
                                int dependent_column_size,
                                float index_map_load_factor, int grid_size,
                                int block_size, float *detail_time);

/**
 * @brief recreate index for a full relation container
 *
 * @param target
 * @param arity
 * @param tuples
 * @param data_row_size
 * @param index_column_size
 * @param dependent_column_size
 * @param index_map_load_factor
 * @param grid_size
 * @param block_size
 */
void reload_full_temp(GHashRelContainer *target, int arity, tuple_type *tuples,
                      tuple_size_t data_row_size,
                      tuple_size_t index_column_size, int dependent_column_size,
                      float index_map_load_factor, int grid_size,
                      int block_size);

enum MonotonicOrder { DESC, ASC, UNSPEC };

/**
 * @brief actual relation class used in semi-naive eval
 *
 */
struct Relation {
    int arity;
    // the first <index_column_size> columns of a relation will be use to
    // build relation index, and only indexed columns can be used to join
    int index_column_size;
    std::string name;

    // the last <dependent_column_size> will be used a dependant columns,
    // these column can be used to store recurisve aggreagtion/choice
    // domain's result, these columns can't be used as index columns
    int dependent_column_size = 0;
    bool index_flag = true;
    bool tmp_flag = false;

    GHashRelContainer *delta;
    GHashRelContainer *newt;
    GHashRelContainer *full;

    tuple_type *tuple_merge_buffer;
    tuple_size_t tuple_merge_buffer_size = 0;
    bool pre_allocated_merge_buffer_flag = true;
    bool fully_disable_merge_buffer_flag = true;
    //

    // delta relation generate in each iteration, all index stripped
    std::vector<GHashRelContainer *> buffered_delta_vectors;

    // reserved properties for monotonic aggregation
    MonotonicOrder monotonic_order = MonotonicOrder::DESC;

    /**
     * @brief store the data in DELTA into full relation (this won't free
     * delta)
     *
     * @param grid_size
     * @param block_size
     */
    void flush_delta(int grid_size, int block_size, float *detail_time);

    void defragement(RelationVersion ver, int grid_size, int block_size);
};

/**
 * @brief load tuples to FULL relation of target relation
 *
 * @param target target relation
 * @param name name of relation
 * @param arity
 * @param data raw flatten tuple need loaded into target relation
 * @param data_row_size number of tuples to load
 * @param index_column_size number of columns used to index
 * @param dependent_column_size
 * @param grid_size
 * @param block_size
 */
void load_relation(Relation *target, std::string name, int arity,
                   column_type *data, tuple_size_t data_row_size,
                   tuple_size_t index_column_size, int dependent_column_size,
                   int grid_size, int block_size, bool tmp_flag = false);

void file_to_buffer(std::string file_path, thrust::host_vector<column_type> &buffer,
                    std::map<column_type, std::string> &string_map);
