#include "saxpy.cu"

int main()
{
    saxpy_kernel<<<1, 1, 0, 0>>>(0, nullptr, nullptr, 0);

    return 0;
}
