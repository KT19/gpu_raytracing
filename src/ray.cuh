#ifndef RAY_CUH
#define RAY_CUH

#include"vec3.cuh"

class ray {
public:
     __host__ __device__ ray() {}
     __host__ __device__ ray(const point3 &origin, const vec3& direction) : orig(origin), dir(direction), tm(0.0f) {}
     __host__ __device__ ray(const point3 &origin, const vec3& direction, float t) : orig(origin), dir(direction), tm(t) {}

     __device__ const point3& origin() const { return orig; }
     __device__ const vec3& direction() const { return dir; }

     __device__ float time() const { return tm; }

     __host__ __device__ point3 at(float t) const {
        return orig + t*dir;
    }

private:
    point3 orig;
    vec3 dir;
    float tm;
};

#endif