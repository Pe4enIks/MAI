#include <stdio.h>
#include <stdlib.h>

int median(int *hist, int cnt) {
    int cumsum = 0;

    for(int i = 0; i < 256; ++i) {
        cumsum += hist[i];
        if(cumsum > cnt / 2) {
            return i;
        }
    }
    return 255;
}

void kernel(uchar4 *inp, uchar4 *out, int w, int h, int r)
{
    uchar4 p;

    int rhist[256], ghist[256], bhist[256];
    int red, green, blue, cnt, x_start, x_end, y_start, y_end;

    for(int y = 0; y < h; ++y) {
        for(int x = 0; x < w; ++x) {
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
                    p = inp[k + m * w];
                    rhist[p.x] += 1;
                    ghist[p.y] += 1;
                    bhist[p.z] += 1;
                    cnt += 1;
                }
            }

            red = median(rhist, cnt);
            green = median(ghist, cnt);
            blue = median(bhist, cnt);

            out[x + y * w] = inp[x + y * w];
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


    uchar4 *out = (uchar4 *)malloc(sizeof(uchar4) * w * h);

    cudaEvent_t start;
    cudaEvent_t stop;

    float time = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    kernel(data, out, w, h, r);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    fp = fopen(out_filename, "wb");

    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(out, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    free(out);

    fprintf(stderr, "time = %f\n", time);
    return 0;
}
