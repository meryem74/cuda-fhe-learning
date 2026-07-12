#include <cstdio>

__global__ void hello()
{
    printf("Merhaba! Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}

int main()
{
    hello<<<2, 4>>>();

    cudaDeviceSynchronize();
    return 0;
}
