#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void reverse(double *init_vct, double *res_vct, int n) {
    for(int i = 0; i < n; ++i)
        res_vct[n - i - 1] = init_vct[i];
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

void init(double *h_init, int n, bool random_flag=true) {
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

    cudaEvent_t start;
    cudaEvent_t stop;

    float time = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    reverse(h_init, h_res, n);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    if(!random_flag) {
        for(int i = 0; i < n; ++i)
            printf("%lf ", h_res[i]);
        printf("\n");
    }

    fprintf(stderr, "time = %f\n", time);

    free(h_init);
    free(h_res);

    return 0;
}