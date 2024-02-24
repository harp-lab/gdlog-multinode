
#include <fstream>
#include <iostream>

#include "exception.cuh"
#include "lie.cuh"
#include "print.cuh"
#include "relation.cuh"
#include "timer.cuh"

const char *test_file_name = "/tmp/test.csv";
const char *test_file_data = "1\tRAX\n2\tRBX\n4125\tRCX";

// void file_to_buffer(std::string file_path, thrust::host_vector<column_type>
// &buffer,
//                     std::map<column_type, std::string> &string_map);

void string_test(int grid_size, int block_size) {
    // write test file data into /tmp/test.csv
    std::ofstream file(test_file_name);
    file << test_file_data;
    file.close();

    Relation *test_relation = new Relation();
    thrust::host_vector<column_type> buffer;
    std::map<column_type, std::string> string_map;
    file_to_buffer(test_file_name, buffer, string_map);
    auto num_of_tuples = buffer.size() / 2;
    load_relation(test_relation, "test", 2, buffer.data(), num_of_tuples, 1, 0,
                  grid_size, block_size);

    // print string_map
    std::cout << "string_map" << std::endl;
    for (auto it = string_map.begin(); it != string_map.end(); it++) {
        std::cout << it->first << " " << it->second << std::endl;
    }

    // print relation
    std::cout << "relation" << std::endl;
    print_tuple_rows(test_relation->full, "test");
}

int main() {
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount,
                           device_id);
    std::cout << "num of sm " << number_of_sm << std::endl;
    std::cout << "using " << EMPTY_HASH_ENTRY << " as empty hash entry"
              << std::endl;
    int block_size, grid_size;
    block_size = 512;
    grid_size = 32 * number_of_sm;

    string_test(grid_size, block_size);
    return 0;
}
