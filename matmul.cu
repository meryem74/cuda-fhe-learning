#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

// Naive GPU matris carpimi: her thread, C'nin TEK BIR hucresini hesaplar.
// C[row][col] = A'nin row. satiri ile B'nin col. sutununun ic carpimi.
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

// Karsilastirma icin klasik CPU versiyonu (3 ic ice dongu).
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
    const int N = 1024;                  // 1024x1024 matris
    size_t bytes = N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);      // GPU sonucu
    float *h_ref = (float*)malloc(bytes);    // CPU sonucu (referans)

    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 10);
        h_B[i] = (float)(rand() % 10);
    }

    // ---------- CPU olcumu ----------
    auto t0 = std::chrono::high_resolution_clock::now();
    matmulCPU(h_A, h_B, h_ref, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---------- GPU hazirlik ----------
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 2D grid: 16x16 = 256 thread'lik block'lar, 64x64 block'luk grid
    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (N + 15) / 16);

    // ---------- cudaEvent ile GPU olcumu ----------
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

    // ---------- Dogrulama ----------
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
