#include <iostream>
#include <stdio.h>
#include <float.h>
#include <vector>
#include <string>
#include <chrono>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#define _USE_MATH_DEFINES
#include <cmath>

#define chrono_cast(start, end) \
    (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000)

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

struct vec3
{
    double x;
    double y;
    double z;
};

struct triangle
{
    vec3 v1;
    vec3 v2;
    vec3 v3;
    vec3 clr;
};

__host__ __device__ vec3 operator+(vec3 lhs, vec3 rhs)
{
    vec3 res = {0.0, 0.0, 0.0};
    res.x = lhs.x + rhs.x;
    res.y = lhs.y + rhs.y;
    res.z = lhs.z + rhs.z;
    return res;
}

__host__ __device__ vec3 operator-(vec3 lhs, vec3 rhs)
{
    vec3 res = {0.0, 0.0, 0.0};
    res.x = lhs.x - rhs.x;
    res.y = lhs.y - rhs.y;
    res.z = lhs.z - rhs.z;
    return res;
}

__host__ __device__ double operator*(vec3 lhs, vec3 rhs)
{
    double res = lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
    return res;
}

__host__ __device__ vec3 operator*(vec3 lhs, double rhs)
{
    vec3 res = {0.0, 0.0, 0.0};
    res.x = lhs.x * rhs;
    res.y = lhs.y * rhs;
    res.z = lhs.z * rhs;
    return res;
}

__host__ __device__ vec3 normalize(vec3 v)
{
    vec3 res = {0.0, 0.0, 0.0};
    res.x = v.x / sqrt(v * v);
    res.y = v.y / sqrt(v * v);
    res.z = v.z / sqrt(v * v);
    return res;
}

__host__ __device__ vec3 prod(vec3 lhs, vec3 rhs)
{
    vec3 res = {0.0, 0.0, 0.0};
    res.x = lhs.y * rhs.z - lhs.z * rhs.y;
    res.y = lhs.z * rhs.x - lhs.x * rhs.z;
    res.z = lhs.x * rhs.y - lhs.y * rhs.x;
    return res;
}

__host__ __device__ vec3 mult(vec3 v1, vec3 v2, vec3 v3, vec3 v4)
{
    vec3 res = {0.0, 0.0, 0.0};
    res.x = v1.x * v4.x + v2.x * v4.y + v3.x * v4.z;
    res.y = v1.y * v4.x + v2.y * v4.y + v3.y * v4.z;
    res.z = v1.z * v4.x + v2.z * v4.y + v3.z * v4.z;
    return res;
}

bool replace(std::string &str,
             const std::string &from,
             const std::string &to)
{
    size_t start = str.find(from);

    if (start == std::string::npos)
        return false;

    str.replace(start, from.length(), to);
    return true;
}

#define raycast()                      \
    vec3 e1 = trgs[k].v2 - trgs[k].v1; \
    vec3 e2 = trgs[k].v3 - trgs[k].v1; \
    vec3 p = prod(dir, e2);            \
    double div = p * e1;               \
    if (fabs(div) < 1e-10)             \
        continue;                      \
    vec3 t = pos - trgs[k].v1;         \
    double u = (p * t) / div;          \
    if (u < 0.0 || u > 1.0)            \
        continue;                      \
    vec3 q = prod(t, e1);              \
    double v = (q * dir) / div;        \
    if (v < 0.0 || v + u > 1.0)        \
        continue;

__host__ __device__ uchar4 ray(vec3 pos,
                               vec3 dir,
                               vec3 light_pos,
                               vec3 light_clr,
                               triangle *trgs,
                               int ssaa_sqrt)
{
    int k_min = -1;
    double ts_min;

    for (int k = 0; k < ssaa_sqrt; ++k)
    {
        raycast();

        double ts = (q * e2) / div;
        if (ts < 0.0)
            continue;

        if (k_min == -1 || ts < ts_min)
        {
            k_min = k;
            ts_min = ts;
        }
    }

    if (k_min == -1)
        return {0, 0, 0, 0};

    pos = dir * ts_min + pos;
    dir = light_pos - pos;

    double size = sqrt(dir * dir);

    dir = normalize(dir);
    for (int k = 0; k < ssaa_sqrt; ++k)
    {
        raycast();

        double ts = (q * e2) / div;
        if (ts > 0.0 && ts < size && k != k_min)
            return {0, 0, 0, 0};
    }

    uchar4 color_min;
    color_min.x = trgs[k_min].clr.x;
    color_min.y = trgs[k_min].clr.y;
    color_min.z = trgs[k_min].clr.z;

    color_min.x *= light_clr.x;
    color_min.y *= light_clr.y;
    color_min.z *= light_clr.z;
    color_min.w = 0;

    return color_min;
}

__global__ void ssaa_gpu(uchar4 *in_data,
                         uchar4 *out_data,
                         int w,
                         int h,
                         int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int y = idy; y < h; y += offsety)
    {
        for (int x = idx; x < w; x += offsetx)
        {
            int4 mid = {0, 0, 0, 0};
            for (int j = 0; j < k; ++j)
            {
                for (int i = 0; i < k; ++i)
                {
                    int ind = k * k * y * w + k * j * w + k * x + i;
                    mid.x += in_data[ind].x;
                    mid.y += in_data[ind].y;
                    mid.z += in_data[ind].z;
                }
            }

            double norm = k * k;
            out_data[x + y * w] = make_uchar4(mid.x / norm,
                                              mid.y / norm,
                                              mid.z / norm,
                                              0);
        }
    }
}

__global__ void gpu_render(vec3 pc,
                           vec3 pv,
                           int w,
                           int h,
                           double angle,
                           uchar4 *data,
                           vec3 light_pos,
                           vec3 light_clr,
                           triangle *trgs,
                           int ssaa_sqrt)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double dz = 1.0 / tan(angle * M_PI / 360.0);

    vec3 bz = normalize(pv - pc);
    vec3 bx = normalize(prod(bz, {0.0, 0.0, 1.0}));
    vec3 by = normalize(prod(bx, bz));

    for (int x = idx; x < w; x += offsetx)
        for (int y = idy; y < h; y += offsety)
        {
            vec3 v = {-1.0 + dw * x, (-1.0 + dh * y) * h / w, dz};
            vec3 dir = mult(bx, by, bz, v);
            data[(h - 1 - y) * w + x] = ray(pc,
                                            normalize(dir),
                                            light_pos,
                                            light_clr,
                                            trgs,
                                            ssaa_sqrt);
        }
}

int gpu(uchar4 *h_data,
        uchar4 *s_data,
        triangle *trgs,
        vec3 pc,
        vec3 pv,
        int w,
        int h,
        int sw,
        int sh,
        double angle,
        vec3 light_pos,
        vec3 light_clr,
        int ssaa_sqrt,
        int k)
{
    uchar4 *d_data;
    uchar4 *d_ssaa_data;
    triangle *d_trgs;

    CSC(cudaMalloc(&d_data, w * h * sizeof(uchar4)));
    CSC(cudaMemcpy(d_data,
                   h_data,
                   w * h * sizeof(uchar4),
                   cudaMemcpyHostToDevice));

    CSC(cudaMalloc(&d_ssaa_data, sw * sh * sizeof(uchar4)));
    CSC(cudaMemcpy(d_ssaa_data,
                   s_data,
                   sw * sh * sizeof(uchar4),
                   cudaMemcpyHostToDevice));

    CSC(cudaMalloc(&d_trgs, ssaa_sqrt * sizeof(triangle)));
    CSC(cudaMemcpy(d_trgs,
                   trgs,
                   ssaa_sqrt * sizeof(triangle),
                   cudaMemcpyHostToDevice));

    gpu_render<<<512, 512>>>(pc,
                             pv,
                             sw,
                             sh,
                             angle,
                             d_ssaa_data,
                             light_pos,
                             light_clr,
                             d_trgs,
                             ssaa_sqrt);

    cudaThreadSynchronize();
    CSC(cudaGetLastError());

    ssaa_gpu<<<512, 512>>>(d_ssaa_data, d_data, w, h, k);

    cudaThreadSynchronize();
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(h_data,
                   d_data,
                   w * h * sizeof(uchar4),
                   cudaMemcpyDeviceToHost));

    CSC(cudaFree(d_data));
    CSC(cudaFree(d_trgs));
    CSC(cudaFree(d_ssaa_data));

    return 0;
}

void ssaa_cpu(uchar4 *in_data,
              uchar4 *out_data,
              int w,
              int h,
              int k)
{
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            int4 mid = {0, 0, 0, 0};
            for (int j = 0; j < k; ++j)
            {
                for (int i = 0; i < k; ++i)
                {
                    int ind = k * k * y * w + k * j * w + k * x + i;
                    mid.x += in_data[ind].x;
                    mid.y += in_data[ind].y;
                    mid.z += in_data[ind].z;
                }
            }
            double norm = k * k;
            out_data[x + y * w] = make_uchar4(mid.x / norm, mid.y / norm, mid.z / norm, 0);
        }
    }
}

void cpu_render(vec3 pc,
                vec3 pv,
                int w,
                int h,
                double angle,
                uchar4 *data,
                vec3 light_pos,
                vec3 light_clr,
                triangle *trgs,
                int ssaa_sqrt)
{
    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double dz = 1.0 / tan(angle * M_PI / 360.0);

    vec3 bz = normalize(pv - pc);
    vec3 bx = normalize(prod(bz, {0.0, 0.0, 1.0}));
    vec3 by = normalize(prod(bx, bz));

    for (int x = 0; x < w; ++x)
        for (int y = 0; y < h; ++y)
        {
            vec3 v = {-1.0 + dw * x, (-1.0 + dh * y) * h / w, dz};
            vec3 dir = mult(bx, by, bz, v);
            data[(h - 1 - y) * w + x] = ray(pc,
                                            normalize(dir),
                                            light_pos,
                                            light_clr,
                                            trgs,
                                            ssaa_sqrt);
        }
}

void cpu(uchar4 *h_data,
         uchar4 *s_data,
         triangle *trgs,
         vec3 pc,
         vec3 pv,
         int w,
         int h,
         int sw,
         int sh,
         double angle,
         vec3 light_pos,
         vec3 light_clr,
         int ssaa_sqrt,
         int k)
{
    cpu_render(pc,
               pv,
               sw,
               sh,
               angle,
               s_data,
               light_pos,
               light_clr,
               trgs,
               ssaa_sqrt);
    ssaa_cpu(s_data, h_data, w, h, k);
}

void create_scene(vec3 p1,
                  vec3 p2,
                  vec3 p3,
                  vec3 p4,
                  vec3 clr,
                  std::vector<triangle> &trgs)
{
    clr.x *= 255.0;
    clr.y *= 255.0;
    clr.z *= 255.0;

    trgs.push_back(triangle{p1, p2, p3, clr});
    trgs.push_back(triangle{p3, p4, p1, clr});
}

void icosa(std::vector<triangle> &trgs,
           double &radius,
           vec3 &centers,
           vec3 &colors)
{
    colors.x *= 255.0;
    colors.y *= 255.0;
    colors.z *= 255.0;

    double arctan = 26.565;
    double angle = M_PI * arctan / 180;
    double segment_angle = M_PI * 72 / 180;
    double current_angle = 0.0;

    std::vector<vec3> points(12);
    points[0] = {0, radius, 0};
    points[11] = {0, -radius, 0};

    for (int i = 1; i < 11; ++i)
    {
        if (i < 6)
        {
            points[i] = {radius * sin(current_angle) * cos(angle),
                         radius * sin(angle),
                         radius * cos(current_angle) * cos(angle)};
            current_angle += segment_angle;
        }
        else
        {
            if (i == 6)
                current_angle = M_PI / 5.0;

            points[i] = {radius * sin(current_angle) * cos(-angle),
                         radius * sin(-angle),
                         radius * cos(current_angle) * cos(-angle)};
            current_angle += segment_angle;
        }
    }

    for (int i = 0; i < 12; ++i)
    {
        points[i].x += centers.x;
        points[i].y += centers.y;
        points[i].z += centers.z;
    }

    trgs.push_back({points[0], points[1], points[2], colors});
    trgs.push_back({points[0], points[2], points[3], colors});
    trgs.push_back({points[0], points[3], points[4], colors});
    trgs.push_back({points[0], points[4], points[5], colors});
    trgs.push_back({points[0], points[5], points[1], colors});

    trgs.push_back({points[1], points[5], points[10], colors});
    trgs.push_back({points[2], points[1], points[6], colors});
    trgs.push_back({points[3], points[2], points[7], colors});
    trgs.push_back({points[4], points[3], points[8], colors});
    trgs.push_back({points[5], points[4], points[9], colors});

    trgs.push_back({points[6], points[7], points[2], colors});
    trgs.push_back({points[7], points[8], points[3], colors});
    trgs.push_back({points[8], points[9], points[4], colors});
    trgs.push_back({points[9], points[10], points[5], colors});
    trgs.push_back({points[10], points[6], points[1], colors});

    trgs.push_back({points[11], points[7], points[6], colors});
    trgs.push_back({points[11], points[8], points[7], colors});
    trgs.push_back({points[11], points[9], points[8], colors});
    trgs.push_back({points[11], points[10], points[9], colors});
    trgs.push_back({points[11], points[6], points[10], colors});
}

void octa(std::vector<triangle> &trgs,
          double &radius,
          vec3 &centers,
          vec3 &colors)
{
    colors.x *= 255.0;
    colors.y *= 255.0;
    colors.z *= 255.0;

    std::vector<vec3> points{{centers.x + radius, centers.y, centers.z},
                             {centers.x - radius, centers.y, centers.z},
                             {centers.x, centers.y + radius, centers.z},
                             {centers.x, centers.y - radius, centers.z},
                             {centers.x, centers.y, centers.z + radius},
                             {centers.x, centers.y, centers.z - radius}};

    trgs.push_back({points[5], points[2], points[0], colors});
    trgs.push_back({points[5], points[0], points[3], colors});
    trgs.push_back({points[5], points[3], points[1], colors});
    trgs.push_back({points[5], points[1], points[2], colors});
    trgs.push_back({points[4], points[3], points[0], colors});
    trgs.push_back({points[4], points[1], points[3], colors});
    trgs.push_back({points[4], points[2], points[1], colors});
    trgs.push_back({points[4], points[0], points[2], colors});
}

void hexa(std::vector<triangle> &trgs,
          double &radius,
          vec3 &centers,
          vec3 &colors)
{
    colors.x *= 255.0;
    colors.y *= 255.0;
    colors.z *= 255.0;

    std::vector<vec3> points{{-1 / sqrt(3), -1 / sqrt(3), -1 / sqrt(3)},
                             {-1 / sqrt(3), -1 / sqrt(3), 1 / sqrt(3)},
                             {-1 / sqrt(3), 1 / sqrt(3), -1 / sqrt(3)},
                             {-1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)},
                             {1 / sqrt(3), -1 / sqrt(3), -1 / sqrt(3)},
                             {1 / sqrt(3), -1 / sqrt(3), 1 / sqrt(3)},
                             {1 / sqrt(3), 1 / sqrt(3), -1 / sqrt(3)},
                             {1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)}};

    for (int i = 0; i < points.size(); ++i)
    {
        points[i].x *= radius;
        points[i].x += centers.x;

        points[i].y *= radius;
        points[i].y += centers.y;

        points[i].z *= radius;
        points[i].z += centers.z;
    }

    trgs.push_back({points[0], points[1], points[3], colors});
    trgs.push_back({points[0], points[2], points[3], colors});
    trgs.push_back({points[1], points[5], points[7], colors});
    trgs.push_back({points[1], points[3], points[7], colors});
    trgs.push_back({points[4], points[5], points[7], colors});
    trgs.push_back({points[4], points[6], points[7], colors});
    trgs.push_back({points[0], points[4], points[6], colors});
    trgs.push_back({points[0], points[2], points[6], colors});
    trgs.push_back({points[0], points[1], points[5], colors});
    trgs.push_back({points[0], points[4], points[5], colors});
    trgs.push_back({points[2], points[3], points[7], colors});
    trgs.push_back({points[2], points[6], points[7], colors});
}

void figures(std::vector<triangle> &trgs,
             double &radius,
             vec3 &centers,
             vec3 &colors,
             std::string &none)
{
    std::cin >> centers.x >> centers.y >> centers.z;
    std::cin >> colors.x >> colors.y >> colors.z;
    std::cin >> radius >> none >> none >> none;
    hexa(trgs, radius, centers, colors);

    std::cin >> centers.x >> centers.y >> centers.z;
    std::cin >> colors.x >> colors.y >> colors.z;
    std::cin >> radius >> none >> none >> none;
    icosa(trgs, radius, centers, colors);

    std::cin >> centers.x >> centers.y >> centers.z;
    std::cin >> colors.x >> colors.y >> colors.z;
    std::cin >> radius >> none >> none >> none;
    octa(trgs, radius, centers, colors);
}

int config()
{
    std::cout << "100\n";
    std::cout << "./out/img_%d.data\n";
    std::cout << "640 480 120\n";
    std::cout << "7.0 3.0 0.0 2.0 1.0 2.0 6.0 1.0 0.0 0.0\n";
    std::cout << "2.0 0.0 0.0 0.5 0.1 1.0 4.0 1.0 0.0 0.0\n";
    std::cout << "4.0 0.0 0.0 1.0 0.7 0.0 1.5 0.0 0.0 0.0\n";
    std::cout << "0.75 1.75 0.0 0.7 0.0 1.0 1.0 0.0 0.0 0.0\n";
    std::cout << "-4.0 -1.5 0.0 0.0 0.5 0.0 0.8 0.0 0.0 0.0\n";
    std::cout << "-10.0 -10.0 -1.0 -10.0 10.0 -1.0 10.0 10.0 ";
    std::cout << "-1.0 10.0 -10.0 -1.0 none 0.3 0.3 0.3 0.25\n";
    std::cout << "1\n";
    std::cout << "25 25 25 1.0 1.0 1.0\n";
    std::cout << "1 5\n";
    return 0;
}

int main(int argc, char *argv[])
{
    std::string arg;
    if (argv[1])
        arg = argv[1];

    if (arg == "--default")
        return config();

    bool on_gpu = argc == 1 || arg == "--gpu";

    std::string none, path;
    int frames_cnt, w, h;
    double angle;
    std::cin >> frames_cnt;
    std::cin >> path;
    std::cin >> w >> h >> angle;

    double roc, zoc, phioc;
    std::cin >> roc >> zoc >> phioc;

    double arc, azc;
    std::cin >> arc >> azc;

    double omegarc, omegazc, omegaphic;
    std::cin >> omegarc >> omegazc >> omegaphic;

    double prc, pzc;
    std::cin >> prc >> pzc;

    double rov, zov, phiov;
    std::cin >> rov >> zov >> phiov;

    double arv, azv;
    std::cin >> arv >> azv;

    double omegarv, omegazv, omegaphiv;
    std::cin >> omegarv >> omegazv >> omegaphiv;

    double prv, pzv;
    std::cin >> prv >> pzv;

    vec3 centers, colors;
    double radius;
    std::vector<triangle> trgs;

    figures(trgs, radius, centers, colors, none);

    vec3 floor_p1, floor_p2, floor_p3, floor_p4;
    std::cin >> floor_p1.x >> floor_p1.y >> floor_p1.z;
    std::cin >> floor_p2.x >> floor_p2.y >> floor_p2.z;
    std::cin >> floor_p3.x >> floor_p3.y >> floor_p3.z;
    std::cin >> floor_p4.x >> floor_p4.y >> floor_p4.z;
    std::cin >> none >> colors.x >> colors.y >> colors.z >> none;
    create_scene(floor_p1, floor_p2, floor_p3, floor_p4, colors, trgs);

    int light_cnt;
    vec3 light_pos, light_clr;
    std::cin >> light_cnt;
    std::cin >> light_pos.x >> light_pos.y >> light_pos.z;
    std::cin >> light_clr.x >> light_clr.y >> light_clr.z;

    int recursion_cnt, ssaa_sqrt;
    std::cin >> recursion_cnt >> ssaa_sqrt;

    triangle *trgs_vct = trgs.data();
    uchar4 *image = new uchar4[w * h * ssaa_sqrt * ssaa_sqrt];
    uchar4 *ssaa_image = new uchar4[w * h * ssaa_sqrt * ssaa_sqrt];

    for (int i = 0; i < frames_cnt; ++i)
    {
        auto start = std::chrono::steady_clock::now();
        double t = i * 2.0 * M_PI / frames_cnt;

        double rc = roc + arc * sin(omegarc * t + prc);
        double zc = zoc + azc * sin(omegazc * t + pzc);
        double phic = phioc + omegaphic * t;
        double rv = rov + arv * sin(omegarv * t + prv);
        double zv = zov + azv * sin(omegazv * t + pzv);
        double phiV = phiov + omegaphiv * t;

        vec3 pc = {rc * cos(phic), rc * sin(phic), zc};
        vec3 pv = {rv * cos(phiV), rv * sin(phiV), zv};

        int trgs_size = trgs.size();

        if (on_gpu)
            gpu(image,
                ssaa_image,
                trgs_vct,
                pc,
                pv,
                w,
                h,
                w * ssaa_sqrt,
                h * ssaa_sqrt,
                angle,
                light_pos,
                light_clr,
                trgs_size,
                ssaa_sqrt);
        else
            cpu(image,
                ssaa_image,
                trgs_vct,
                pc,
                pv,
                w,
                h,
                w * ssaa_sqrt,
                h * ssaa_sqrt,
                angle,
                light_pos,
                light_clr,
                trgs_size,
                ssaa_sqrt);

        auto end = std::chrono::steady_clock::now();
        double time = chrono_cast(start, end);

        std::cout << i << "\t" << time << "\t";
        std::cout << w * h * ssaa_sqrt * ssaa_sqrt << "\n";

        std::string epoch = std::to_string(i);
        std::string filename = path;
        replace(filename, "%d", epoch);

        FILE *fp = fopen(filename.c_str(), "wb");
        fwrite(&w, sizeof(int), 1, fp);
        fwrite(&h, sizeof(int), 1, fp);
        fwrite(image, sizeof(uchar4), w * h, fp);
        fclose(fp);
    }

    delete[] image;
    delete[] ssaa_image;

    return 0;
}