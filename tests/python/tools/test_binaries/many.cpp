#include "many.cu"

int main()
{
    say_hi<<<1, 1, 0, 0>>>();

    vector_atomic_add_42<<<1, 1, 0, 0>>>(nullptr, nullptr, nullptr, 0);

    return 0;
}
