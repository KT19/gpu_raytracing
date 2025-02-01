#ifndef RENDER_CORE_CU
#define RENDER_CORE_CU

#include<cstdio>

__device__ float linear_to_gamma(float linear_component) {
    return linear_component > 0.0f ? sqrtf(linear_component) : 0.0f;
}

//write color utility functions
__device__ color ray_trace(const ray& r, hittable_list* world, curandState* local_rand_state, int max_depth) {
    ray cur_ray = r;
    color cur_attenuation = color(1.0f, 1.0f, 1.0f);
    color color_from_emission = color(0, 0, 0);    
    color background = color(0, 0, 0); //Scene background color

    for(int i = 0;i < max_depth;i++) {
        hit_record rec;
     
        if(world->hit(cur_ray, interval(0.001f, infinity), rec)) {
            ray scattered;
            color attenuation;
            color_from_emission += rec.mat->emit(rec, local_rand_state);
            if(rec.mat->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return cur_attenuation*(color_from_emission); // no scatter
            }
        } else {
                    
            return cur_attenuation*(background);
        }
    }

    //terminate due to the recursion
    return vec3(0.0f, 0.0f, 0.0f); //exceeded recursion
}

//rendering function
__global__ void render(
    uchar4* pixels,
    Camera* d_camera,
    float time, 
    hittable_list* world,
    curandState* rand_state
    ) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;


    if((x >= d_camera->image_width) || (y >= d_camera->image_height)) return;
    //construct ray
    vec3 pixel_center = d_camera->pixel00_loc + (x * d_camera->pixel_delta_u) + (y * d_camera->pixel_delta_v);
    vec3 ray_direction = pixel_center - d_camera->center;

    int pixel_idx = (x + y*d_camera->image_width);
    curandState local_rand_state = rand_state[pixel_idx];

    color pixel_color = color(0.0f, 0.0f, 0.0f);
    int num_per_pixel_ratio = 1;

    // used to check the visual appearance  
    // if(float(d_camera->image_width/3.0f) < x && x <= float(d_camera->image_width*2.0f)/3.0f ) {
    //      num_per_pixel_ratio = 8; 
    // } else if(float(d_camera->image_width*2.0f)/3.0f < x) {
    //     num_per_pixel_ratio = 64;
    // }

    for(int i = 0;i < d_camera->samples_per_pixel * num_per_pixel_ratio;i++) {
        ray r = d_camera->get_ray(x, y, &local_rand_state);
        
        pixel_color += ray_trace(r, world, &local_rand_state, d_camera->max_depth) * d_camera->pixel_samples_scale / num_per_pixel_ratio;
    }

    rand_state[pixel_idx] = local_rand_state;

    uchar4 pixel;    
    const interval intensity(0.0f, 1.0f);
    
    pixel.x = static_cast<unsigned char>(255.99 * intensity.clamp(linear_to_gamma(pixel_color.x())));
    pixel.y = static_cast<unsigned char>(255.99 * intensity.clamp(linear_to_gamma(pixel_color.y())));
    pixel.z = static_cast<unsigned char>(255.99 * intensity.clamp(linear_to_gamma(pixel_color.z())));
    pixel.w = 255;

    pixels[pixel_idx] = pixel;
}


#endif