#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

void swap_kernel(double *out, int i, int j, int n)
{
    double tmp;

    for (int k = 0; k < 2 * n; k += 1)
    {
        tmp = out[i + k * n];
        out[i + k * n] = out[j + k * n];
        out[j + k * n] = tmp;
    }
}

void nullification_down_kernel(double *out, int i, int n)
{
    int startx = i + 1;
    int endx = n;
    int starty = i + 1;
    int endy = 2 * n;

    for (int x = startx; x < endx; x += 1)
        for (int y = starty; y < endy; y += 1)
            out[x + y * n] = -out[x + i * n] / out[i * (n + 1)] * out[i + y * n] + out[x + y * n];
}

void nullification_up_kernel(double *out, int i, int n)
{
    int startx = i - 1;
    int endx = 0;
    int starty = n;
    int endy = 2 * n;

    for (int x = startx; x >= endx; x -= 1)
        for (int y = starty; y < endy; y += 1)
            out[x + y * n] = -out[x + i * n] / out[i * (n + 1)] * out[i + y * n] + out[x + y * n];
}

void divide_by_diagonal_kernel(double *out, int n)
{
    int startx = 0;
    int endx = n;
    int starty = n;
    int endy = 2 * n;

    for (int x = startx; x < endx; x += 1)
        for (int y = starty; y < endy; y += 1)
            out[x + y * n] /= out[x * (n + 1)];
}

int get_max(double *data, int i, int n)
{
    int start = i * (n + 1);
    int end = (i + 1) * n;

    double maxv = -DBL_MAX;
    int ind = -1;

    for (int k = start; k < end; ++k)
    {
        if (data[k] > maxv)
        {
            maxv = data[k];
            ind = k;
        }
    }

    return ind;
}

int main()
{
    bool visual_debug_flag = false;
    bool debug_flag = false;
    bool time_flag = true;
    bool matrix_size_flag = false;

    int n;
    FILE *fp;

    fscanf(stdin, "%d", &n);
    double *h_data = (double *)malloc(2 * n * n * sizeof(double));

    if (matrix_size_flag)
        fprintf(stderr, "%d\n", n);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            fscanf(stdin, "%lf", &h_data[i + j * n]);

    if (debug_flag)
    {
        fp = fopen("./debug.txt", "w");
        fprintf(fp, "%d ", n);
        for (int i = 0; i < n * n; ++i)
            fprintf(fp, "%lf ", h_data[i]);
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i != j)
                h_data[i + j * n + n * n] = 0.0;
            else
                h_data[i + j * n + n * n] = 1.0;
        }
    }

    if (visual_debug_flag)
    {
        fprintf(stderr, "\ninit matrix by rows\n");
        for (int i = 0; i < 2 * n * n; ++i)
        {
            if (i % n == 0 && i != 0)
                fprintf(stderr, "\n");
            fprintf(stderr, "%lf ", h_data[i]);
        }
        fprintf(stderr, "\n");
    }

    cudaEvent_t start;
    cudaEvent_t stop;

    float time = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    for (int i = 0; i < n - 1; ++i)
    {
        int j = get_max(h_data, i, n) - i * n;

        if (i != j)
            swap_kernel(h_data, i, j, n);

        nullification_down_kernel(h_data, i, n);
    }

    for (int i = n - 1; i > 0; --i)
        nullification_up_kernel(h_data, i, n);

    divide_by_diagonal_kernel(h_data, n);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    if (visual_debug_flag)
    {
        fprintf(stderr, "\nresult matrix by rows\n");
        for (int i = 0; i < 2 * n * n; ++i)
        {
            if (i % n == 0 && i != 0)
                fprintf(stderr, "\n");
            fprintf(stderr, "%lf ", h_data[i]);
        }
        fprintf(stderr, "\n");
    }

    if (debug_flag)
    {
        for (int i = n * n; i < 2 * n * n; ++i)
            fprintf(fp, "%lf ", h_data[i]);
        fclose(fp);
    }

    if (!time_flag)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
            {
                if (j != n - 1)
                    fprintf(stdout, "%.10lf ", h_data[i + j * n + n * n]);
                else
                    fprintf(stdout, "%.10lf\n", h_data[i + j * n + n * n]);
            }

    free(h_data);

    if (time_flag)
        fprintf(stderr, "\ntime = %f\n", time);
    return 0;
}
