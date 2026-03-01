struct Base {
    virtual ~Base() = default;
    __device__ virtual bool guard() = 0;
    __device__ virtual void work(float&) = 0;
};

struct Derived : public Base {
    __device__ bool guard() override {
        return true;
    }
    __device__ void work(float& value) override {
        value += 42.;
    }
};

__global__ void kernel(Base* const ptr, float* const data) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (ptr->guard())
        ptr->work(data[index]);
}
