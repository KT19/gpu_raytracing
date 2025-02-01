#ifndef CAMERA_CU
#define CAMERA_CU

#include"hittable.cuh"
#include"material.cuh"

class Camera {
public:
    int image_height;
    int image_width;
    point3 center;
    point3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    float pixel_samples_scale;
    int samples_per_pixel;
    int max_depth;
    float vfov = 40.0f;
    point3 lookfrom;
    point3 lookat;
    vec3 vup = vec3(0, 1, 0);

    float defocus_angle = 0.0f;
    float focus_dist = 10.0f; //should be replaced with focal length

    __device__ Camera() {};
    
    __device__ Camera(int image_height, int image_width, point3 lookfrom, point3 lookat, int ns = 10, int max_depth=3) {
        this->image_height = image_height; 
        this->image_width = image_width; 
        this->aspect_ratio = float(image_width)/float(image_height);
        this->lookfrom = lookfrom;
        this->lookat = lookat;
        this->center = lookfrom;
        this->samples_per_pixel = ns;
        this->pixel_samples_scale = 1.0f / float(samples_per_pixel);
        this->max_depth = max_depth;

        this->initialize();
    }

    __device__ ray get_ray(int i, int j, curandState *local_rand_state) const {
        vec3 offset = sample_square(local_rand_state);
        vec3 pixel_sample = pixel00_loc + ((i + offset.x())*pixel_delta_u) + ((j + offset.y())*pixel_delta_v);

        vec3 ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample(local_rand_state);
        vec3 ray_direction = pixel_sample - ray_origin;
        float ray_time = random_float(local_rand_state);

        return ray(ray_origin, ray_direction, ray_time);
    }

    __device__ point3 defocus_disk_sample(curandState* local_rand_state) const {
        auto p = random_in_unit_disk(local_rand_state);
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

private:
    float aspect_ratio;
    vec3 u, v, w;
    vec3 defocus_disk_u;
    vec3 defocus_disk_v;

    __device__ void initialize(void) {
        float theta = degrees_to_radians(vfov);
        float h = tanf(theta);
        float viewport_height = h * focus_dist;
        float viewport_width = viewport_height * float(image_width)/float(image_height);
        
        //calculate the u, v, w unit basis vectors
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        //viewport edges
        vec3 viewport_u = viewport_width*u;
        vec3 viewport_v = viewport_height*v;
        //horizontal and vertical delta vectors from pixel to pixel
        pixel_delta_u = viewport_u / float(image_width);
        pixel_delta_v = viewport_v / float(image_height);
        //location of the upper-left
        vec3 viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

        //Calculate the camera defocus disk basis vectors
        auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle/2.0f));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;

        return;
    }

    __device__ vec3 sample_square(curandState *local_rand_state) const {
        return vec3(random_float(local_rand_state) - 0.5f, random_float(local_rand_state) - 0.5f, 0.0f);
    }
};

#endif