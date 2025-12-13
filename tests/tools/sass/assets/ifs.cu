__global__ void ifs(int* __restrict__ const data, const unsigned int size)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (data[idx] > 0) {
            data[idx] -= idx * 42;
        } else {
            data[idx] += idx * 666;
        }
    } else if (size > 0) {
        data[0] += 1;
    }
}
