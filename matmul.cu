#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

__global__ void matmulGPU(const float* A, const float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void matmulCPU(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main()
{
    const int N = 1024;                  
    size_t bytes = N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);      
    float *h_ref = (float*)malloc(bytes);    

    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 10);
        h_B[i] = (float)(rand() % 10);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    matmulCPU(h_A, h_B, h_ref, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (N + 15) / 16);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmulGPU<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);


    bool ok = true;
    for (int i = 0; i < N * N; i++)
        if (fabs(h_C[i] - h_ref[i]) > 1e-3) { ok = false; break; }

    printf("Dogrulama : %s\n", ok ? "BASARILI" : "HATALI");
    printf("CPU suresi: %.2f ms\n", cpu_ms);
    printf("GPU suresi: %.2f ms\n", gpu_ms);
    printf("Hizlanma  : %.1fx\n", cpu_ms / gpu_ms);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
