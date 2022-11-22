#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#define CSC(call)                                                 \
	do                                                            \
	{                                                             \
		cudaError_t res = call;                                   \
		if (res != cudaSuccess)                                   \
		{                                                         \
			fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
					__FILE__, __LINE__, cudaGetErrorString(res)); \
			exit(0);                                              \
		}                                                         \
	} while (0)

struct comparator
{
	__host__ __device__ bool operator()(double lhs, double rhs)
	{
		return fabs(lhs) < fabs(rhs);
	}
};

__global__ void swap_kernel(double *out, int i, int j, int n)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	double tmp;

	for (int k = idx; k < 2 * n; k += offsetx)
	{
		tmp = out[i + k * n];
		out[i + k * n] = out[j + k * n];
		out[j + k * n] = tmp;
	}
}

__global__ void nullification_down_kernel(double *out, int i, int n)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

	int startx = idx + i + 1;
	int endx = n;
	int starty = idy + i + 1;
	int endy = 2 * n;

	for (int x = startx; x < endx; x += offsetx)
		for (int y = starty; y < endy; y += offsety)
			out[x + y * n] = -out[x + i * n] / out[i * (n + 1)] * out[i + y * n] + out[x + y * n];
}

__global__ void nullification_up_kernel(double *out, int i, int n)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

	int startx = i - idx - 1;
	int endx = 0;
	int starty = n + idy;
	int endy = 2 * n;

	for (int x = startx; x >= endx; x -= offsetx)
		for (int y = starty; y < endy; y += offsety)
			out[x + y * n] = -out[x + i * n] / out[i * (n + 1)] * out[i + y * n] + out[x + y * n];
}

__global__ void divide_by_diagonal_kernel(double *out, int n)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

	int startx = idx;
	int endx = n;
	int starty = n + idy;
	int endy = 2 * n;

	for (int x = startx; x < endx; x += offsetx)
		for (int y = starty; y < endy; y += offsety)
			out[x + y * n] /= out[x * (n + 1)];
}

int main()
{
	bool visual_debug_flag = false;
	bool debug_flag = false;
	bool time_flag = true;
	bool matrix_size_flag = false;

	int n;
	comparator comp;
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

	double *d_data;
	CSC(cudaMalloc(&d_data, 2 * n * n * sizeof(double)));
	CSC(cudaMemcpy(d_data, h_data, 2 * n * n * sizeof(double), cudaMemcpyHostToDevice));

	cudaEvent_t start;
	cudaEvent_t stop;

	float time = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	for (int i = 0; i < n - 1; ++i)
	{
		thrust::device_ptr<double> p_arr = thrust::device_pointer_cast(d_data);
		thrust::device_ptr<double> res = thrust::max_element(p_arr + i * (n + 1), p_arr + (i + 1) * n, comp);
		int j = (int)(res - p_arr) - i * n;

		if (i != j)
			swap_kernel<<<512, 512>>>(d_data, i, j, n);

		nullification_down_kernel<<<dim3(32, 16), dim3(32, 16)>>>(d_data, i, n);
	}

	for (int i = n - 1; i > 0; --i)
		nullification_up_kernel<<<dim3(32, 16), dim3(32, 16)>>>(d_data, i, n);

	divide_by_diagonal_kernel<<<dim3(32, 16), dim3(32, 16)>>>(d_data, n);

	cudaEventRecord(stop, 0);

	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);

	cudaEventDestroy(stop);
	cudaEventDestroy(start);

	CSC(cudaGetLastError());

	CSC(cudaMemcpy(h_data, d_data, 2 * n * n * sizeof(double), cudaMemcpyDeviceToHost));

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

	CSC(cudaFree(d_data));
	free(h_data);

	if (time_flag)
		fprintf(stderr, "\ntime = %f\n", time);
	return 0;
}
