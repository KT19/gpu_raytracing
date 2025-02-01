#ifndef INTERVAL_CUH
#define INTERVAL_CUH

class interval{
public:
    float min, max;
    
    __host__ __device__ interval() : min(+infinity), max(-infinity) {}
    __host__ __device__ interval(float mi, float ma) : min(mi), max(ma) {}
    __host__ __device__ interval(const interval& a, const interval& b) {
        min = a.min <= b.min ? a.min : b.min;
        max = a.max <= b.max ? a.max : b.max;
    }

    __device__ float size() const {
        return max - min;
    }

    __device__ bool contains(float x) const {
        return min <= x && x <= max;
    }

    __device__ bool surrounds(float x) const {
        return min < x && x < max;
    }

    __device__ float clamp(float x) const {
        if(x < min) return min;
        if(x > max) return max;
        return x;
    }

    __device__ interval expand(float delta) const {
        auto padding = delta / 2;
        return interval(min - padding, max + padding);
    }

    static const interval empty, universe;
};

const interval interval::empty = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity);

#endif