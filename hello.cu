#include <cstdio>

// __global__ = "bu fonksiyon GPU'da çalışir, CPU'dan cagrilir" demek.
// Bu fonksiyonlara "kernel" denir.
__global__ void hello()
{
    // Her thread kendi kimligini yazdirir.
    printf("Merhaba! Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}

int main()
{
    // <<<2, 4>>> : 2 block, her block'ta 4 thread => toplam 8 thread.
    hello<<<2, 4>>>();

    // GPU asenkron calisir; bitmesini beklemezsek program cikti gormeden biter.
    cudaDeviceSynchronize();
    return 0;
}
