

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main() {

    thrust::device_vector<int> vec(3);
    vec[0] = 1;
    vec[1] = 2;
    vec[2] = 3;

    thrust::host_vector<int> h_vec = vec;

    for (int i = 0; i < h_vec.size(); i++) {
        std::cout << h_vec[i] << std::endl;
    }
    
    return 0;
}

