#ifndef VEC3_CUH
#define VEC3_CUH

class vec3 {
public:
    float e[3];

    __host__ __device__ vec3(): e{0, 0, 0} {}
    __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }

    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]);}
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; }

    __host__ __device__ inline vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];

        return *this;
    }

    __host__ __device__ inline vec3& operator*=(float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;

        return *this;
    }

    __host__ __device__ inline vec3& operator*=(const vec3& v) {
        e[0] *= v[0];
        e[1] *= v[1];
        e[2] *= v[2];

        return *this;
    }

    __host__ __device__ inline vec3& operator/=(float t) {
        return *this *= 1.0f/t;
    }

    __host__ __device__ inline float length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    __host__ __device__ inline float length() const {
        return sqrtf(length_squared());
    }

    __host__ __device__ inline bool near_zero() const {
        float s = 1e-6f;

        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }

    __device__ static vec3 random(curandState* local_rand_state) {
        return vec3(random_float(local_rand_state), random_float(local_rand_state), random_float(local_rand_state));
    }

    __device__ static vec3 random(float min, float max, curandState* local_rand_state) {
        return vec3(random_float(min, max, local_rand_state), random_float(min, max, local_rand_state), random_float(min, max, local_rand_state));
    }
};

//alias
using point3 = vec3;

//functions
__host__ __device__ inline vec3 operator+(const vec3& u, float v) {
    return vec3(u.e[0] + v, u.e[1] + v, u.e[2] + v);
}
__host__ __device__ inline vec3 operator+(float v, const vec3& u) {
    return u+v;
}
__host__ __device__ inline vec3 operator-(const vec3& u, float v) {
    return vec3(u.e[0] - v, u.e[1] - v, u.e[2] - v);
}
__host__ __device__ inline vec3 operator-(float v, const vec3& u) {
    return vec3(v - u.e[0], v - u.e[1], v - u.e[2]);
}
__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, float t) {
    return t*u;
}

__host__ __device__ inline vec3 operator/(const vec3& u, float t) {
    return (1.0f/t)*u;
}

__host__ __device__ inline float dot(const vec3& u, const vec3& v) {
    return u.e[0]*v.e[0] + u.e[1]*v.e[1] + u.e[2]*v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(
        u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]
    );
}

__host__ __device__ inline vec3 unit_vector(const vec3& v) {
   
    return v / v.length();
}

__device__ inline vec3 random_unit_vector(curandState* local_rand_state) {
    while(true) {
        auto p = vec3::random(-1.0f, 1.0f, local_rand_state);
        auto lensq = p.length_squared();
        if(1e-5f < lensq && lensq <= 1.0f) {
            return p / sqrtf(lensq);
        }
    }
}

__device__ inline vec3 random_on_hemisphere(const vec3& normal, curandState* local_rand_state) {
    vec3 on_unit_sphere = random_unit_vector(local_rand_state);

    return dot(on_unit_sphere, normal) > 0.0f ? on_unit_sphere : -on_unit_sphere;
}

__device__ inline vec3 random_in_unit_disk(curandState* local_rand_state) {
    while(true) {
        auto p = vec3(random_float(-1.0f, 1.0f, local_rand_state), random_float(-1.0f, 1.0f, local_rand_state), 0.0f);
        if(p.length_squared() < 1)
            return p;
    }
}

__device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.0f*dot(v, n)*n;
}

__device__ inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    float cos_theta = fmin(dot(-uv, n), 1.0f);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -sqrt(fabs(1.0f - r_out_perp.length_squared()))*n;
    return r_out_perp + r_out_parallel;
}

#endif