#ifndef HITTABLE_LIST_CUH
#define HITTABLE_LIST_CUH

#include"hittable.cuh"
#include"primitive.cuh"

class hittable_list {
public:
    __device__ hittable_list() {}
    __device__ hittable_list(primitive* obj_list, int n) {
        objects = obj_list;
        obj_num = n;
    }
    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* rand_state = NULL) const {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = ray_t.max;

        for(int i = 0;i < obj_num;i++) {
            const auto object = objects[i];
     
            if(object.hit(r, interval(ray_t.min, closest_so_far), temp_rec, rand_state)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
                rec.p = rec.p;
            }
        }

        return hit_anything;
    }
    __device__ inline primitive* get_list_ptr() { return objects; }
    __device__ inline int get_object_num() { return obj_num; }
private:
    primitive* objects;
    int obj_num;
};

#endif