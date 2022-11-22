#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define CSC(call)                                                     \
do {                                                                  \
    cudaError_t res = call;                                           \
    if(res != cudaSuccess) {                                          \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",              \
                        __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(0);                                                      \
    }                                                                 \
} while(0)

struct vec3 {
	double r, g, b;
};

struct matrix {
	double arr[3][3];
};

vec3 avg[32];
matrix cov[32];
double logv[32];
__constant__ vec3 d_avg[32];
__constant__ matrix d_cov[32];
__constant__ double d_logv[32];

__global__ void kernel(uchar4 *out, int w, int h, int nc)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;

    uchar4 p;
	int argmax;
	double maxv, v;
	vec3 curr_sub0, curr_sub1;

	for(int x = idx; x < w * h; x += offsetx) {
		argmax = -1;
		maxv = -DBL_MAX;

		for(int cls = 0; cls < nc; ++cls) {
			p = out[x];
			curr_sub0.r = p.x - d_avg[cls].r;
			curr_sub0.g = p.y - d_avg[cls].g;
			curr_sub0.b = p.z - d_avg[cls].b;
			curr_sub1.r = curr_sub0.r * d_cov[cls].arr[0][0] + curr_sub0.g * d_cov[cls].arr[1][0] + curr_sub0.b * d_cov[cls].arr[2][0];
			curr_sub1.g = curr_sub0.r * d_cov[cls].arr[0][1] + curr_sub0.g * d_cov[cls].arr[1][1] + curr_sub0.b * d_cov[cls].arr[2][1];
			curr_sub1.b = curr_sub0.r * d_cov[cls].arr[0][2] + curr_sub0.g * d_cov[cls].arr[1][2] + curr_sub0.b * d_cov[cls].arr[2][2];
			v = -(curr_sub0.r * curr_sub1.r + curr_sub0.g * curr_sub1.g + curr_sub0.b * curr_sub1.b) - d_logv[cls];
			if(v > maxv) {
				maxv = v;
				argmax = cls;
			}
		}

		out[x].w = argmax;
	}
}

__host__ void inverse(matrix *matr, int cls) {
	matrix minor_matr;

    double minor1 =  (matr[cls].arr[1][1] * matr[cls].arr[2][2] - matr[cls].arr[2][1] * matr[cls].arr[1][2]);
    double minor2 = -(matr[cls].arr[1][0] * matr[cls].arr[2][2] - matr[cls].arr[2][0] * matr[cls].arr[1][2]);
    double minor3 =  (matr[cls].arr[1][0] * matr[cls].arr[2][1] - matr[cls].arr[2][0] * matr[cls].arr[1][1]);
    double minor4 = -(matr[cls].arr[0][1] * matr[cls].arr[2][2] - matr[cls].arr[2][1] * matr[cls].arr[0][2]);
    double minor5 =  (matr[cls].arr[0][0] * matr[cls].arr[2][2] - matr[cls].arr[2][0] * matr[cls].arr[0][2]);
    double minor6 = -(matr[cls].arr[0][0] * matr[cls].arr[2][1] - matr[cls].arr[2][0] * matr[cls].arr[0][1]);
    double minor7 =  (matr[cls].arr[0][1] * matr[cls].arr[1][2] - matr[cls].arr[1][1] * matr[cls].arr[0][2]);
    double minor8 = -(matr[cls].arr[0][0] * matr[cls].arr[1][2] - matr[cls].arr[1][0] * matr[cls].arr[0][2]);
    double minor9 =  (matr[cls].arr[0][0] * matr[cls].arr[1][1] - matr[cls].arr[1][0] * matr[cls].arr[0][1]);

	minor_matr.arr[0][0] = minor1;
	minor_matr.arr[0][1] = minor2;
	minor_matr.arr[0][2] = minor3;
	minor_matr.arr[1][0] = minor4;
	minor_matr.arr[1][1] = minor5;
	minor_matr.arr[1][2] = minor6;
	minor_matr.arr[2][0] = minor7;
	minor_matr.arr[2][1] = minor8;
	minor_matr.arr[2][2] = minor9;

	double dval = (matr[cls].arr[0][0] * minor1 + matr[cls].arr[0][1] * minor2 + matr[cls].arr[0][2] * minor3);

    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            minor_matr.arr[i][j] /= dval;
			matr[cls].arr[i][j] = minor_matr.arr[i][j];
        }
    }
}

__host__ void preprocessing(uchar4 *img, int nc, int w, vec3 *avg, matrix *cov, double *logv) {
	int npj;
	uchar4 p;
	vec3 sub;
	double det;

	for(int cls = 0; cls < nc; ++cls) {
		scanf("%d", &npj);
		int *x_arr = (int *)malloc(npj * sizeof(int));
		int *y_arr = (int *)malloc(npj * sizeof(int));

		avg[cls].r = 0.0;
		avg[cls].g = 0.0;
		avg[cls].b = 0.0;

		for(int i = 0; i < 3; ++i) {
			for(int j = 0; j < 3; ++j) {
				cov[cls].arr[i][j] = 0.0;
			}
		}

		for(int np = 0; np < npj; ++np) {
			scanf("%d %d", &x_arr[np], &y_arr[np]);
			p = img[x_arr[np] + y_arr[np] * w];
			avg[cls].r += p.x;
			avg[cls].g += p.y;
			avg[cls].b += p.z;
		}

		avg[cls].r /= npj;
		avg[cls].g /= npj;
		avg[cls].b /= npj;

		for(int np = 0; np < npj; ++np) {
			p = img[x_arr[np] + y_arr[np] * w];
			sub.r = p.x - avg[cls].r;
			sub.g = p.y - avg[cls].g;
			sub.b = p.z - avg[cls].b;
			cov[cls].arr[0][0] += sub.r * sub.r;
			cov[cls].arr[0][1] += sub.r * sub.g;
			cov[cls].arr[0][2] += sub.r * sub.b;
			cov[cls].arr[1][0] += sub.g * sub.r;
			cov[cls].arr[1][1] += sub.g * sub.g;
			cov[cls].arr[1][2] += sub.g * sub.b;
			cov[cls].arr[2][0] += sub.b * sub.r;
			cov[cls].arr[2][1] += sub.b * sub.g;
			cov[cls].arr[2][2] += sub.b * sub.b;
		}

		if(npj > 1) {
			for(int i = 0; i < 3; ++i) {
				for(int j = 0; j < 3; ++j) {
					cov[cls].arr[i][j] /= (npj - 1);
				}
			}
		}

		det = 0.0;
		det += cov[cls].arr[0][0] * cov[cls].arr[1][1] * cov[cls].arr[2][2];
		det -= cov[cls].arr[2][0] * cov[cls].arr[1][1] * cov[cls].arr[0][2];
		det += cov[cls].arr[1][0] * cov[cls].arr[2][1];
		det += cov[cls].arr[0][1] * cov[cls].arr[1][2];
		det -= cov[cls].arr[2][1] * cov[cls].arr[1][2];
		det -= cov[cls].arr[1][0] * cov[cls].arr[0][1];

		if(det < 0.0)
			det *= -1.0;

		logv[cls] = log(det);
		inverse(cov, cls);

		free(x_arr);
		free(y_arr);
	}
}

int main() {
    int w, h, nc;
    char inp_filename[256], out_filename[256];
    
    scanf("%s", inp_filename);
    scanf("%s", out_filename);
    scanf("%d", &nc);

    FILE *fp = fopen(inp_filename, "rb");
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

	preprocessing(data, nc, w, avg, cov, logv);

    CSC(cudaMemcpyToSymbol(d_avg, avg, 32 * sizeof(vec3)));
    CSC(cudaMemcpyToSymbol(d_cov, cov, 32 * sizeof(matrix)));
    CSC(cudaMemcpyToSymbol(d_logv, logv, 32 * sizeof(double)));

    uchar4 *d_out;
    CSC(cudaMalloc(&d_out, sizeof(uchar4) * w * h));
	CSC(cudaMemcpy(d_out, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;

    float time = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    kernel<<<512, 512>>>(d_out, w, h, nc);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, d_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
    CSC(cudaFree(d_out));

    fp = fopen(out_filename, "wb");

    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    fprintf(stderr, "time = %f\n", time);
    return 0;
}
