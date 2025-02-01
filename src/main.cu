#ifndef MAIN_CU
#define MAIN_CU

//#include"scene_simple_light.cu"
#include"scene_cornell_box.cu"

int main() {
    int image_width = 600;
    int image_height = 600;
    srand(time(0));
    
    //create_simple_light(image_width, image_height);
    create_cornell_box(image_width, image_height);

    return 0;
}

#endif