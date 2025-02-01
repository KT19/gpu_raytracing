#ifndef SCENE_CORNELL_BOX
#define SCENE_CORNELL_BOX

#include"scene_init.cuh"

//camera initialization
__global__ void camera_init(Camera* d_camera, int image_height, int image_width, int ns, int max_depth) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        *d_camera = Camera(image_height, image_width, point3(278, 278, -800), point3(278, 278, 0), ns, max_depth);
    }
}

//texture solid initialization
__global__ void init_texture_on_device(texture* d_texture) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        //red
        d_texture[0].set_texture_type(SOLID_COLOR);
        d_texture[0].set_solid_color(color(.65f, .05f, .05f));
        //white
        d_texture[1].set_texture_type(SOLID_COLOR);
        d_texture[1].set_solid_color(color(.73f, .73, .73f));
        //green
        d_texture[2].set_texture_type(SOLID_COLOR);
        d_texture[2].set_solid_color(color(.12f, .45f, .15f));
        //light
        d_texture[3].set_texture_type(SOLID_COLOR);
        d_texture[3].set_solid_color(color(15, 15, 15));
    }
}

//allocate materials on the host
void init_materials_on_host(int num_objs, material* d_materials) {
    material* h_materials = new material[num_objs];
    int cur_index = 0;

    h_materials[cur_index++] = material(DIFFUSE, color(.65f, .05f, .05f), 0.0f, 1.0f);
    h_materials[cur_index++] = material(DIFFUSE, color(.73f, .73f, .73f), 0.0f, 1.0f); //noise
    h_materials[cur_index++] = material(DIFFUSE, color(.12f, .45f, .15f), 0.0f, 1.0f); //noise
    h_materials[cur_index++] = material(DIFFUSE_LIGHT, color(15, 15, 15), 0.0f, 1.0f); //diffuse_light

    checkCudaErrors(cudaMemcpy(d_materials, h_materials, num_objs*sizeof(material), cudaMemcpyHostToDevice));

    delete [] h_materials;

    return;
}

//allocate sphers on host using materials defined in the cuda
void init_world_on_host(int num_objs, primitive* d_primitives) {
    primitive* h_primitives = new primitive[num_objs];
    int cur_index = 0;

    h_primitives[cur_index].set_primitive_type(QUAD);
    h_primitives[cur_index++].set_quad(point3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555));

    h_primitives[cur_index].set_primitive_type(QUAD);
    h_primitives[cur_index++].set_quad(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555));

    h_primitives[cur_index].set_primitive_type(QUAD);
    h_primitives[cur_index++].set_quad(point3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105));
    
    h_primitives[cur_index].set_primitive_type(QUAD);
    h_primitives[cur_index++].set_quad(point3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555));
    
    h_primitives[cur_index].set_primitive_type(QUAD);
    h_primitives[cur_index++].set_quad(point3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555));
    
    h_primitives[cur_index].set_primitive_type(QUAD);
    h_primitives[cur_index++].set_quad(point3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0));
    
    h_primitives[cur_index].set_primitive_type(BOX);
    h_primitives[cur_index].set_box(point3(0, 0, 0), point3(165, 330, 165));
    //geometry manipulation
    h_primitives[cur_index].set_rotate(15.0f);
    h_primitives[cur_index++].set_translate(vec3(265, 0, 295));

    h_primitives[cur_index].set_primitive_type(BOX);
    h_primitives[cur_index].set_box(point3(0, 0, 0), point3(165, 165, 165));
    //geometry manipulation
    h_primitives[cur_index].set_rotate(-18.0f);
    h_primitives[cur_index++].set_translate(vec3(130, 0, 65));

    checkCudaErrors(cudaMemcpy(d_primitives, h_primitives, num_objs*sizeof(primitive), cudaMemcpyHostToDevice));
    
    delete [] h_primitives;

    return;
}

//Apply material to objects
__global__ void apply_materials_on_device(int num_objs, primitive* d_primitives, material* d_materials, texture* d_texture) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        //apply texture
        d_materials[3].set_tex(&d_texture[3]);
        //apply materials
        d_primitives[0].apply_material(&d_materials[2]);
        d_primitives[1].apply_material(&d_materials[0]);
        d_primitives[2].apply_material(&d_materials[3]);
        d_primitives[3].apply_material(&d_materials[1]);
        d_primitives[4].apply_material(&d_materials[1]);
        d_primitives[5].apply_material(&d_materials[1]);
        d_primitives[6].apply_material(&d_materials[1]);
        d_primitives[7].apply_material(&d_materials[1]);
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

void create_cornell_box(int image_width, int image_height) {
    printf("hello create cornell box\n");
    //camera setting
    Camera* d_camera;
    printf("start camera init...\n");
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera)));
    camera_init<<<1, 1>>>(d_camera, image_height, image_width, 64, 5);
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
    int num_objs = 8;
    primitive* d_primitives;
    printf("start init world on host...\n");
    checkCudaErrors(cudaMalloc((void**)&d_primitives, num_objs*sizeof(primitive)));
    //create world
    init_world_on_host(num_objs, d_primitives);
    printf("end init world on host...\n");

    //setup texture
    texture* d_texture;

    checkCudaErrors(cudaMalloc((void**)&d_texture, 4*sizeof(texture)));
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