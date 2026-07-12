#include <cstdio>
#include <cstdlib>


__global__ void vectorAdd(const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread sayisi n'den fazla olabilir; tasanlari engelle.
    if (i < n)
        c[i] = a[i] + b[i];
}

int main()
{
    const int N = 1 << 20;               
    size_t bytes = N * sizeof(float);

    
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;  // yukari yuvarlama: 4096 block
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (h_c[i] != 3.0f) { ok = false; break; }
    printf(ok ? "DOGRU: tum elemanlar 3.0\n" : "HATA!\n");

    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
