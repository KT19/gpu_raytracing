#ifndef RTWEEKEND_CUH
#define RTWEEKEND_CUH

#include<cmath>
#include<iostream>
#include<limits>

using namespace std;

float rand_host(float rmin, float rmax) {
    return rmin + (rmax - rmin)*((float)rand() / (RAND_MAX));
}

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535f;

__device__ inline float mix(float a, float b, float t) {
    return t*a + (1.0f - t)*b;
}

__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}
__device__ inline float fract(float x) {
    return x - floor(x);
}
__device__ float random_float(curandState* local_rand_state) {
    return curand_uniform(local_rand_state);
}

__device__ float random_float(float min, float max, curandState* local_rand_state) {
    return min + curand_uniform(local_rand_state)*(max - min);
}

__device__ int random_int(int min, int max, curandState* local_rand_state) {
    return int(random_float(min, max+1, local_rand_state));
}

int random_int_host(int min, int max) {
    return int(rand_host(min, max+1));
}


#include"color.cu"
#include"interval.cuh"
#include"ray.cuh"
#include"vec3.cuh"
#include"error_handling.cu"

#endif