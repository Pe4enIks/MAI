#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void kernel(double *init_vct, double *res_vct, int n) {
    int ind = blockDim.x * blockIdx.x + threadIdx.x;
	int step = blockDim.x * gridDim.x;

	for(int i = ind; i < n; i += step) {
		res_vct[i] = init_vct[n - i - 1];
    }
}

void random(double *vct, int n) {
    double rand_state = 7.0;

    for(int i = 0; i < n; ++i) {
        vct[i] = ((double)rand() / (double)(RAND_MAX)) * rand_state;
    }
}

void scan(double *vct, int n) {
    double val = 0.0;
    for(int i = 0; i < n; ++i) {
        scanf("%lf", &val);
        vct[i] = val;
    }
}

void init(double *h_init, int n, bool random_flag) {
    if(random_flag)
        random(h_init, n);
    else
        scan(h_init, n);
}

int main() {
    bool random_flag = false;
    int n = 0;

    if(!random_flag)
        scanf("%d", &n);

    double *h_init = (double*)malloc(sizeof(double) * n);
    double *h_res = (double*)malloc(sizeof(double) * n);

    init(h_init, n, random_flag);

    double *d_init, *d_res;

    cudaMalloc(&d_init, sizeof(double) * n);
    cudaMemcpy(d_init, h_init, sizeof(double) * n, cudaMemcpyHostToDevice);

    cudaMalloc(&d_res, sizeof(double) * n);
    cudaMemcpy(d_res, h_res, sizeof(double) * n, cudaMemcpyHostToDevice);

    cudaEvent_t start;
    cudaEvent_t stop;

    float time = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    kernel<<<32, 32>>>(d_init, d_res, n);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    cudaMemcpy(h_res, d_res, sizeof(double) * n, cudaMemcpyDeviceToHost);

    if(!random_flag) {
        for(int i = 0; i < n; ++i)
            printf("%lf ", h_res[i]);
        printf("\n");
    }

    fprintf(stderr, "time = %f\n", time);

    cudaFree(d_init);
    cudaFree(d_res);

    free(h_init);
    free(h_res);

    return 0;
}
