#ifndef SCENE_SIMPLE_LIGHT
#define SCENE_SIMPLE_LIGHT

#include"scene_init.cuh"

//camera initialization
__global__ void camera_init(Camera* d_camera, int image_height, int image_width, int ns, int max_depth) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        *d_camera = Camera(image_height, image_width, point3(26, 3, 6), point3(0, 2, 0), ns, max_depth);
    }
}

//texture noise initilization
__global__ void init_noise_on_device(procedural_perlin* d_noise, texture* d_texture) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        d_texture[0].set_texture_type(PROCECURAL_NOISE);
        d_texture[0].set_procedural_noise(d_noise);
    }

    return;
}

//texture solid initialization
__global__ void init_texture_on_device(texture* d_texture) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        d_texture[1].set_texture_type(SOLID_COLOR);
        d_texture[1].set_solid_color(color(4, 4, 4));
    }
}

//allocate materials on the host
void init_materials_on_host(int num_objs, material* d_materials) {
    material* h_materials = new material[num_objs];
    int cur_index = 0;

    h_materials[cur_index++] = material(DIFFUSE, color(1.0f, 0.2f, 0.2f), 0.0f, 1.0f); //noise
    h_materials[cur_index++] = material(DIFFUSE, color(0.2f, 1.0f, 0.2f), 0.0f, 1.0f); //noise
    h_materials[cur_index++] = material(DIFFUSE_LIGHT, color(4, 4, 4), 0.0f, 1.0f); //diffuse_light

    checkCudaErrors(cudaMemcpy(d_materials, h_materials, num_objs*sizeof(material), cudaMemcpyHostToDevice));

    delete [] h_materials;

    return;
}

//allocate sphers on host using materials defined in the cuda
void init_world_on_host(int num_objs, primitive* d_primitives) {
    primitive* h_primitives = new primitive[num_objs];
    int cur_index = 0;

    h_primitives[cur_index].set_primitive_type(SPHERE);
    h_primitives[cur_index++].set_stationary_sphere(point3(0, -1000, 0), 1000);
    
    h_primitives[cur_index].set_primitive_type(SPHERE);
    h_primitives[cur_index++].set_stationary_sphere(point3(0, 2, 0), 2);
    
    h_primitives[cur_index].set_primitive_type(QUAD);
    h_primitives[cur_index++].set_quad(point3(3, 1, -2), vec3(2, 0, 0), vec3(0, 2, 0));
    
    checkCudaErrors(cudaMemcpy(d_primitives, h_primitives, num_objs*sizeof(primitive), cudaMemcpyHostToDevice));
    
    delete [] h_primitives;

    return;
}

//Apply material to objects
__global__ void apply_materials_on_device(int num_objs, primitive* d_primitives, material* d_materials, texture* d_texture) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        //apply texture
        d_materials[0].set_tex(&d_texture[0]);
        d_materials[1].set_tex(&d_texture[0]);
        d_materials[2].set_tex(&d_texture[1]);
        for(int i = 0;i < num_objs;i++) {
            d_primitives[i].apply_material(&d_materials[i]);
        }
    }
    
    return;
}

__global__ void render_init(int image_width, int image_height, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i >= image_width) || (j >= image_height)) return;

    int pixel_index = j * image_width + i;
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);    
}

__global__ void create_world(int num_objs, hittable_list* d_world, primitive* d_primitives) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        *d_world = hittable_list(d_primitives, num_objs);
    }
}

__global__ void free_world(hittable_list* d_world) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        delete d_world;
    }
}

void create_simple_light(int image_width, int image_height) {
    printf("hello create simple light\n");
    //camera setting
    Camera* d_camera;
    printf("start camera init...\n");
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera)));
    camera_init<<<1, 1>>>(d_camera, image_height, image_width, 50, 5);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    printf("end camera init...\n");

    printf("start opengl init...\n");
    DrawPixels dp(image_width, image_height);
    dp.Init();
    printf("end opengl init...\n");

    //memory in the cuda
    size_t fb_size = image_height * image_width * sizeof(uchar4);
    uchar4 *fb;

    //allocate on the device memory
    checkCudaErrors(cudaMalloc((void**)&fb, fb_size));
    
    //curand set
    curandState* d_rand_state;
    //malloc
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, image_height*image_width*sizeof(curandState)));

    //Allocate sphere capacity
    //setup object numbers
    int num_objs = 3;
    primitive* d_primitives;
    printf("start init world on host...\n");
    checkCudaErrors(cudaMalloc((void**)&d_primitives, num_objs*sizeof(primitive)));
    //create world
    init_world_on_host(num_objs, d_primitives);
    printf("end init world on host...\n");

    //setup texture
    procedural_perlin* d_noise;
    texture* d_texture;

    printf("start init noise...\n");
    checkCudaErrors(cudaMalloc((void**)&d_noise, sizeof(procedural_perlin)));
    checkCudaErrors(cudaMalloc((void**)&d_texture, 2*sizeof(texture)));
    init_noise_on_device<<<1,1>>>(d_noise, d_texture);
    printf("end init noise...\n");
    printf("start set texture...\n");
    init_texture_on_device<<<1,1>>>(d_texture);
    printf("end set texture...\n");
    
    //Allocate material capacity
    printf("start init materials...\n");
    material* d_materials;
    checkCudaErrors(cudaMalloc((void**)&d_materials, num_objs*sizeof(material)));
    //create material
    init_materials_on_host(num_objs, d_materials);
    printf("end init materials...\n");

    //apply material
    printf("start apply materials...\n");
    apply_materials_on_device<<<1, 1>>>(num_objs, d_primitives, d_materials, d_texture);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    printf("end apply materials...\n");

    //hittable_list world
    hittable_list* d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable_list))); 

    printf("start create world...\n");
    create_world<<<1, 1>>>(num_objs, d_world, d_primitives);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    printf("end create world...\n");


    //thread size
    int tx = 16;
    int ty = 16;

    dim3 blocks((image_width + tx - 1)/tx, (image_height + ty - 1)/ty);
    dim3 threads(tx, ty);

    //setup
    printf("start render init...\n");
    render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    printf("end render init...\n");

    dp.Loop(blocks, threads, fb, d_camera, d_world, d_rand_state);

    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_materials));
    checkCudaErrors(cudaFree(d_primitives));
}

#endif