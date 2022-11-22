#include <stdio.h>
#include <stdlib.h>

#define CSC(call)                                                     \
do {                                                                  \
    cudaError_t res = call;                                           \
    if(res != cudaSuccess) {                                          \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",              \
                        __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(0);                                                      \
    }                                                                 \
} while(0)

texture<uchar4, 2, cudaReadModeElementType> tex;

__device__ int median(int *hist, int cnt) {
    int cumsum = 0;

    for(int i = 0; i < 256; ++i) {
        cumsum += hist[i];
        if(cumsum > cnt / 2) {
            return i;
        }
    }
    return 255;
}

__global__ void kernel(uchar4 *out, int w, int h, int r)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    uchar4 p;

    int rhist[256], ghist[256], bhist[256];
    int red, green, blue, cnt, x_start, x_end, y_start, y_end;

    for(int y = idy; y < h; y += offsety) {
        for(int x = idx; x < w; x += offsetx) {
            cnt = 0;

            for(int i = 0; i < 256; ++i) {
                rhist[i] = 0;
                ghist[i] = 0;
                bhist[i] = 0;
            }

            x_start = (x - r >= 0) ? x - r : 0;
            x_end = (x + r < w) ? x + r : w - 1;
            y_start = (y - r >= 0) ? y - r : 0;
            y_end = (y + r < h) ? y + r : h - 1;

            for(int m = y_start; m <= y_end; ++m) {
                for(int k = x_start; k <= x_end; ++k) {
                    p = tex2D(tex, k, m);
                    rhist[p.x] += 1;
                    ghist[p.y] += 1;
                    bhist[p.z] += 1;
                    cnt += 1;
                }
            }

            red = median(rhist, cnt);
            green = median(ghist, cnt);
            blue = median(bhist, cnt);

            out[x + y * w] = tex2D(tex, x, y);
            out[x + y * w].x = red;
            out[x + y * w].y = green;
            out[x + y * w].z = blue;
        }
    }
}


int main() {
    int w, h, r;
    char inp_filename[256], out_filename[256];
    
    scanf("%s", inp_filename);
    scanf("%s", out_filename);
    scanf("%d", &r);

    FILE *fp = fopen(inp_filename, "rb");
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

    tex.normalized = false;
    tex.filterMode = cudaFilterModePoint;
    tex.channelDesc = ch;
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;

    CSC(cudaBindTextureToArray(tex, arr, ch));

    uchar4 *d_out;
    CSC(cudaMalloc(&d_out, sizeof(uchar4) * w * h));

    cudaEvent_t start;
    cudaEvent_t stop;

    float time = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    kernel<<<dim3(16, 16), dim3(32, 8)>>>(d_out, w, h, r);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    CSC(cudaGetLastError());
    CSC(cudaMemcpy(data, d_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
    CSC(cudaUnbindTexture(tex));

    CSC(cudaFreeArray(arr));
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
