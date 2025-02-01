#ifndef TEXTURE_CUH
#define TEXTURE_CUH
#include"color.cu"
#include"rtw_stb_image.cuh"
#include"perlin.cuh"

#define SOLID_COLOR (0)
#define IMG_TEXTURE (1)
#define PROCECURAL_NOISE (2)

class texture {
public:
    __device__ __host__ texture() {}
    __device__ void set_texture_type(int t) {
        type = t;
    }
    //solid color texture
    __device__ void set_solid_color(color a) {
        albedo = a;
    }
    //image texture
    __device__ __host__ void set_image_texture_info(int imh, int imw, int bps) {
        image_height = imh; 
        image_width = imw;
        bytes_per_scanline = bps;
    }
    //image texture
    __device__ void register_data(unsigned char* data) {
        d_bdata = data;
    }

    //procedural noise
    __device__ void set_procedural_noise(procedural_perlin* noise_registered) {
        d_p_noise = noise_registered;
    }

    __device__ color value(float u, float v, const point3& p) const {
        if(type == SOLID_COLOR) {

            return albedo;

        } else if(type == IMG_TEXTURE) {
            if(image_height <= 0) return color(0.0f, 1.0f, 1.0f);

            u = interval(0, 1).clamp(u);
            v = 1.0f - interval(0, 1).clamp(v);

            auto i = int(u * image_width);
            auto j = int(v * image_height);
            auto pixel = pixel_data(d_bdata, i, j, image_width, image_height, bytes_per_scanline);

            auto color_scale = 1.0f / 255.0f;

            return color(color_scale*pixel[0], color_scale*pixel[1], color_scale*pixel[2]);
        } else if(type == PROCECURAL_NOISE) {
            return color(0.5f, 0.5f, 0.5f) * (1.0f + sin(p.z() + 10.0f*d_p_noise->noise(p)));
        }

        return color(1.0f, 1.0f, 0.0f);
    }
private:
    //texture type
    int type;
    //for solid color
    color albedo;
    //for image texture
    unsigned char* d_bdata;
    int image_height;
    int image_width;
    int bytes_per_scanline;
    //for procedural perlin
    procedural_perlin* d_p_noise;
};

#endif