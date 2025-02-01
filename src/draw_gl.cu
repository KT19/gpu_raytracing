#ifndef DRAW_GL_CU
#define DRAW_GL_CU

#include<iostream>
#include<cuda_runtime.h>
#include<cuda_gl_interop.h>
#include<GLFW/glfw3.h>

#include"vec3.cuh"
#include"camera.cu"
#include"render_core.cu"

//callback
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if(key == GLFW_KEY_Q && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    
    return;
}

class DrawPixels {
private:
    GLFWwindow* window;
    GLuint texture;
    cudaArray* texturePtr;
    cudaGraphicsResource* cudaResource;
    int WIDTH, HEIGHT;

    bool initOpenGL(void);
    void drawCUDA(uchar4* devicePtr);
    void renderGL(void);

public:
    //constructor
    DrawPixels(int width, int height) : WIDTH(width), HEIGHT(height) {}
    
    //cleanup
    ~DrawPixels() {
        //Unmap
        cudaGraphicsUnmapResources(1, &cudaResource, 0);
        //cleanup
        cudaGraphicsUnregisterResource(cudaResource);
        glDeleteTextures(1, &texture);
        glfwDestroyWindow(window);
        glfwTerminate();
    }
    void Init(void);
    void Loop(dim3 blocks, dim3 threads, uchar4* devicePtr, Camera* d_camera, hittable_list* world, curandState* rand_state);
};

void DrawPixels::Init(void) {
    initOpenGL();
    
    //Set the key callback
    glfwSetKeyCallback(window, keyCallback);

    //Init CUDA
    cudaGraphicsGLRegisterImage(&cudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    //mapping
    cudaGraphicsMapResources(1, &cudaResource, 0);
    cudaGraphicsSubResourceGetMappedArray(&texturePtr, cudaResource, 0, 0);

    return;
}

bool DrawPixels::initOpenGL(void) {
    if(!glfwInit()) {
        std::cerr<<"Failed to initialize GLFW"<<std::endl;
        return false;
    }
    //create
    window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGLwindow", NULL, NULL);
    if(!(window)) {
        std::cerr<<"Failed to create GLFW window"<<std::endl;
        glfwTerminate();
        return false;
    }

    //context current
    glfwMakeContextCurrent(window);

    //Generate OpenGL texture
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return true;
}

void DrawPixels::Loop(dim3 blocks, dim3 threads, uchar4* devicePtr, Camera* d_camera, hittable_list* world, curandState* rand_state) {
    //game loop
    while(!glfwWindowShouldClose(window)) {
        float time = (float)glfwGetTime();

        render<<<blocks, threads>>>(devicePtr, d_camera, time, world, rand_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        drawCUDA(devicePtr);
        renderGL();

        //Double buffering
        glfwSwapBuffers(window);
        //event poll
        glfwPollEvents();
    }

}

//copy device contents to textures
void DrawPixels::drawCUDA(uchar4* devicePtr) {
    cudaMemcpy2DToArray(texturePtr, 0, 0, devicePtr, WIDTH*sizeof(uchar4), WIDTH*sizeof(uchar4), HEIGHT, cudaMemcpyDeviceToDevice);

    return;
}

//OpenGL rendering
void DrawPixels::renderGL(void) {
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);

    glBegin(GL_QUADS);
    //Fill in the entire screen
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
    glEnd();

    glDisable(GL_TEXTURE_2D);

    return;
}

#endif