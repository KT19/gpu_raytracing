#ifndef PERLIN_CUH
#define PERLIN_CUH

class procedural_perlin {
public:
    __device__ __host__ procedural_perlin() {}

    __device__ float noise(const point3& p) {
        float val = 0.0f;
        float amp = 1.0f;
        float freq = 1.0f;
        float G = 0.5f;
        int numCount = 5;

        for(int i = 0;i < numCount;i++) {
            val += amp * perlinNoise(p * freq);
            amp *= G;
            freq *= 2.0f;
        }

        return fabs(val);
    }
private:
    __device__ float perlinNoise(const vec3& p) {
        vec3 gridId = vec3(mod289(floor(p.x())), mod289(floor(p.y())), mod289(floor(p.z())));
        vec3 gridUV = vec3(fract(p.x()), fract(p.y()), fract(p.z()));

        //define box corner
        vec3 n010 = permute(gridId + vec3(0.0f, 1.0f, 0.0f));
        vec3 n110 = permute(gridId + vec3(1.0f, 1.0f, 0.0f));
        vec3 n000 = permute(gridId + vec3(0.0f, 0.0f, 0.0f));
        vec3 n100 = permute(gridId + vec3(1.0f, 0.0f, 0.0f));

        vec3 n011 = permute(gridId + vec3(0.0f, 1.0f, 1.0f));
        vec3 n111 = permute(gridId + vec3(1.0f, 1.0f, 1.0f));
        vec3 n001 = permute(gridId + vec3(0.0f, 0.0f, 1.0f));
        vec3 n101 = permute(gridId + vec3(1.0f, 0.0f, 1.0f));

        //random grad in each box
        float grad010 = grad(int(n010.x()), gridUV - vec3(0.0f, 1.0f, 0.0f));//randomGradient(ctl);
        float grad110 = grad(int(n110.x()), gridUV - vec3(1.0f, 1.0f, 0.0f));//randomGradient(ctr);
        float grad000 = grad(int(n000.x()), gridUV - vec3(0.0f, 0.0f, 0.0f));//randomGradient(cbl);
        float grad100 = grad(int(n100.x()), gridUV - vec3(1.0f, 0.0f, 0.0f));//randomGradient(cbr);
        float grad011 = grad(int(n011.x()), gridUV - vec3(0.0f, 1.0f, 1.0f));//randomGradient(ftl);
        float grad111 = grad(int(n111.x()), gridUV - vec3(1.0f, 1.0f, 1.0f));//randomGradient(ftr);
        float grad001 = grad(int(n001.x()), gridUV - vec3(0.0f, 0.0f, 1.0f));//randomGradient(fbl);
        float grad101 = grad(int(n101.x()), gridUV - vec3(1.0f, 0.0f, 1.0f));//randomGradient(fbr);


        gridUV = gridUV*gridUV*gridUV*(10.0f + gridUV*(-15.0f + gridUV*6.0f));

        //interpolation
        float nx00 = mix(grad000, grad100, gridUV.x());
        float nx01 = mix(grad001, grad101, gridUV.x());
        float nx10 = mix(grad010, grad110, gridUV.x());
        float nx11 = mix(grad011, grad111, gridUV.x());
        
        float nxy0 = mix(nx00, nx10, gridUV.y());
        float nxy1 = mix(nx01, nx11, gridUV.y());

        return mix(nxy0, nxy1, gridUV.z());
    }

    __device__ float mod289(float x) {
        return x - floor(x / 289.0f) * 289.0f;
    }

    __device__ vec3 permute(const vec3& p) {
        return vec3(
            mod289((34.0f*p.x() + 1.0f)*p.x()),
            mod289((34.0f*p.y() + 1.0f)*p.y()),
            mod289((34.0f*p.z() + 1.0f)*p.z())
            );
    }

    __device__ float grad(int hash, const vec3& p) {
        int h = hash & 15;
        float u = h < 8 ? p.x() : p.y();
        float v = h < 4 ? p.y() : (h == 12 || h == 14 ? p.x() : p.z());
        
        return ((h&1) == 0 ? u : -u) + ((h&2) == 0 ? v : -v);
    }

    __device__ vec3 randomGradient(const vec3& p) {
        float x = dot(p, vec3(123.4f, 234.5f, 345.6f));
        float y = dot(p, vec3(234.5f, 345.6f, 456.7f));
        float z = dot(p, vec3(345.6f, 456.7f, 567.8f));

        float s = 43758.5453123f;

        vec3 gradient = vec3(sin(x), sin(y), sin(z))*s;
        
        return vec3(sin(gradient.x()), sin(gradient.y()), sin(gradient.z()));
    }
};

#endif