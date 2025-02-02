#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#define DIFFUSE (0)
#define METAL (1)
#define GLASS (2)
#define DIFFUSE_LIGHT (3)
#define ISOTROPIC (4)

#include"hittable.cuh"
#include"texture.cuh"

class material {
public:
    __host__ __device__ ~material() {};

    __host__ __device__ material() {};

    __host__ __device__ material(int mat_t, const color& a, float f, float refr_i) : mat_type(mat_t), albedo(a), fuzz(f), refraction_index(refr_i), use_tex(false) {};


    __device__ void set_tex(texture* tex_ptr) {
        tex = tex_ptr;
        use_tex = true;
    }

    __device__ bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state
    ) const {
        if(mat_type == DIFFUSE_LIGHT) return false;
        
        attenuation = use_tex ? tex->value(rec.u, rec.v, rec.p) : albedo;
        
        if(mat_type == DIFFUSE) {
            auto scatter_direction = rec.normal + random_unit_vector(local_rand_state);
            
            if(scatter_direction.near_zero())
                scatter_direction = rec.normal;

            scattered = ray(rec.p, scatter_direction, r_in.time());

            return true;

        } else if(mat_type == METAL) {
            vec3 reflected = reflect(r_in.direction(), rec.normal);
            reflected += fuzz * random_unit_vector(local_rand_state);
            scattered = ray(rec.p, reflected, r_in.time());

            return (dot(scattered.direction(), rec.normal) > 0);

        } else if(mat_type == GLASS) {
            float ri = rec.front_face ? (1.0f / refraction_index) : refraction_index;

            vec3 unit_direction = unit_vector(r_in.direction());

            float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0f);
            float sin_theta = sqrt(1.0f - cos_theta*cos_theta);

            bool cannot_refract = ri * sin_theta > 1.0f;
            vec3 direction;

            if(cannot_refract || (reflectance(cos_theta, ri) > random_float(local_rand_state))) {
                direction = reflect(unit_direction, rec.normal);
            } else {
                direction = refract(unit_direction, rec.normal, ri);
            }

            scattered = ray(rec.p, direction, r_in.time());
            
            return true;
        } else if(mat_type == ISOTROPIC) {
            scattered = ray(rec.p, random_unit_vector(local_rand_state), r_in.time());

            return true;
        }

        return false;
    }
    __device__ color emit(const hit_record& rec, curandState* local_rand_state) const {
        if(mat_type == DIFFUSE_LIGHT) {
            return tex->value(rec.u, rec.v, rec.p);
        }

        return color(0, 0, 0);
    }

    int mat_type;
private:
    texture* tex;
    color albedo;
    float fuzz;
    float refraction_index;
    bool use_tex;

    __device__ static float reflectance(float cosine, float refraction_index) {
        float r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
        r0 = r0 * r0;

        return r0 + (1.0f - r0)*powf((1.0f - cosine), 5.0f);
    }
};


#endif