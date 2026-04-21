#include "saxpy.cu"

int main() {
    saxpy_kernel<<<1, 1, 0, 0>>>(0, 0, nullptr, nullptr);

    return 0;
}
