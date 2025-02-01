#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include<curand_kernel.h>

class material;

class hit_record {
public: 
    point3 p;
    vec3 normal;
    material* mat;
    float t;
    float u; //texture-coordinate
    float v; //texture-coordinate
    bool front_face;

    __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


#endif