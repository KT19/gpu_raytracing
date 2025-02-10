#ifndef PRIMITIVE_CUH
#define PRIMITIVE_CUH

#include"hittable.cuh"
#include"material.cuh"
#include"ray.cuh"

#define SPHERE (1)
#define QUAD (2)
#define BOX (3)
#define CONSTANT_MEDIUM (4)

struct Quad {
    point3 Q;
    vec3 u, v;
    vec3 w;
    vec3 normal;
    float D;
};

class primitive {
public:
    //common
    __host__ primitive() {}
    __host__ void set_primitive_type(int p) { type = p; }
    __device__ void apply_material(material* m) {
        mat = m;
    }
    
    //sphere
    //moving
    __host__ __device__ void set_moving_sphere(const point3& center1, const point3 center2, float _radius) {
        center = ray(center1, center2 - center1); 
        radius = fmax(0.0f, _radius); 
    }
    //sphere
    __host__ __device__ void set_stationary_sphere(const point3& _center, float _radius) {
        center = ray(_center, vec3(0, 0, 0));
        radius = fmax(0.0f, _radius);
    }

    __device__ static void get_sphere_uv(const point3& p, float &u, float &v) {
        //p: a given point on the sphere of radius one, centered at the origin
        //u: returned value [0, 1] of angle
        //v: returned value [0, 1] from Y=-1, Y=+1
        float theta = std::acos(-p.y());
        float phi = std::atan2(-p.z(), p.x()) + pi;

        u = phi / (2*pi);
        v = theta / pi;
    }

    //quad
    __host__ __device__ void set_quad(const point3& Q, const vec3& u, const vec3&v) {
        set_quad_primitive(_q, Q, u, v);
    }

    //box
    __host__ __device__ void set_box(const point3& a, const point3& b) {
        //opposite vertices
        auto min3 = point3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
        auto max3 = point3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));
        //box size
        auto dx = vec3(max3.x() - min3.x(), 0, 0);
        auto dy = vec3(0, max3.y() - min3.y(), 0);
        auto dz = vec3(0, 0, max3.z() - min3.z());

        //construct box
        //Note: qb[0]: front, qb[1]: right, qb[2]: back
        //qb[3]: left, qb[4]: top, qb[5]: bottom
        set_quad_primitive(qb[0], point3(min3.x(), min3.y(), max3.z()), dx, dy);
        set_quad_primitive(qb[1], point3(max3.x(), min3.y(), max3.z()), -dz, dy);
        set_quad_primitive(qb[2], point3(max3.x(), min3.y(), min3.z()), -dx, dy);
        set_quad_primitive(qb[3], point3(min3.x(), min3.y(), min3.z()), dz, dy);
        set_quad_primitive(qb[4], point3(min3.x(), max3.y(), max3.z()), dx, -dz);
        set_quad_primitive(qb[5], point3(min3.x(), min3.y(), min3.z()), dx, dz);
    }
    
    //convert to constant medium
    __host__ __device__ void make_constant_medium(float d) {
        neg_inv_density = -1.0f/d;
        medium_flag = true;
    }
    
    //geometry manipulation
    __host__ __device__ void set_translate(const vec3& _t) {
        trans = _t;
    }
    __host__ __device__ void set_rotate(float _angle) {
        float rad = degrees_to_radians(_angle);
        sin_theta = sinf(rad);
        cos_theta = cosf(rad);
    }

    //rotation    
    __device__ ray rotate_y(const ray& r) const {
        auto origin = point3(
            (cos_theta*r.origin().x()) - (sin_theta*r.origin().z()),
            (r.origin().y()),
            (sin_theta*r.origin().x()) + (cos_theta*r.origin().z())
        );
        
        auto direction = point3(
            (cos_theta*r.direction().x()) - (sin_theta*r.direction().z()),
            (r.direction().y()),
            (sin_theta*r.direction().x()) + (cos_theta*r.direction().z())
        );

        return ray(origin, direction, r.time());
    }
    __device__ void rotate_y_post_process(hit_record& rec) const {
        rec.p = point3(
            (cos_theta*rec.p.x()) + (sin_theta*rec.p.z()),
            (rec.p.y()),
            (-sin_theta*rec.p.x()) + (cos_theta*rec.p.z())
        );
        
        rec.normal = point3(
            (cos_theta*rec.normal.x()) + (sin_theta*rec.normal.z()),
            (rec.normal.y()),
            (-sin_theta*rec.normal.x()) + (cos_theta*rec.normal.z())
        );
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState *rand_state) const {
        if(medium_flag) { //i.e., some medium case
            hit_record rec1, rec2;

            //check in
            if(!_hit(r, interval(-infinity, +infinity), rec1, rand_state)) return false;
           
            //check out
            if(!_hit(r, interval(rec1.t+0.01f, infinity), rec2, rand_state)) return false;

            if(rec1.t < ray_t.min) rec1.t = ray_t.min;
            if(rec2.t > ray_t.max) rec2.t = ray_t.max;

            if(rec1.t >= rec2.t) return false;

            if(rec1.t < 0) rec1.t = 0;

            auto ray_length = r.direction().length();
            auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
            auto hit_distance = neg_inv_density * logf(random_float(rand_state));
            
            if(hit_distance > distance_inside_boundary) return false;

            rec.t = rec1.t + hit_distance / ray_length;
            rec.p = r.at(rec.t);

            rec.normal = vec3(1, 0, 0);
            rec.front_face = true;
            rec.mat = mat;

            return true;
        }

        return _hit(r, ray_t, rec, rand_state);
    }

    __device__ bool _hit(const ray& _r, interval ray_t, hit_record& rec, curandState *rand_state = NULL) const {
        //apply translation
        ray trans_r(_r.origin() - trans, _r.direction(), _r.time());
        //apply rotation
        ray r = rotate_y(trans_r);
        bool hit_anything = false;
        
        if(type == SPHERE) {
            point3 cur_center = center.at(r.time());
            vec3 oc = cur_center - r.origin();
            float a = r.direction().length_squared();
            float h = dot(r.direction(), oc);
            float c = oc.length_squared() - radius*radius;
            
            float discriminant = h*h - a * c;
            if(discriminant < 0) return false;
        
            float sqrtd = sqrt(discriminant);

            float root = (h - sqrtd) / a;

            if(!ray_t.surrounds(root)) {
        
                root = (h + sqrtd) / a;
                if(!ray_t.surrounds(root)) {
                    return false;
                }
            }

            rec.t = root;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - cur_center) / radius;
    
            rec.set_face_normal(r, outward_normal);
            get_sphere_uv(outward_normal, rec.u, rec.v);
            rec.mat = mat;

            hit_anything = true;

        } else if(type == QUAD) {
            hit_anything = quad_hit(_q, r, ray_t, rec, rand_state);
        } else if(type == BOX) {
            hit_record temp_rec;
            float closest_so_far = ray_t.max;

            //since box consists of 6 quads
            for(int i = 0;i < 6;i++) {
                if(quad_hit(qb[i],r, interval(ray_t.min, closest_so_far), temp_rec, rand_state) && temp_rec.t < closest_so_far) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }
        }
        //post process
        if(hit_anything) {
            rotate_y_post_process(rec);
            rec.p = rec.p + trans;
        }

        return hit_anything;
    }

private:
    //common
    int type;
    material* mat;
    vec3 trans = vec3(0, 0, 0);
    float sin_theta = 0, cos_theta = 1;
    
    //for sphere
    ray center;
    float radius;

    //for quad
    struct Quad _q;

    //for box
    struct Quad qb[6]; //used to define box

    //for constant medium
    float neg_inv_density;
    bool medium_flag = false;

    //giving quad structure to be set
    __host__ __device__ void set_quad_primitive(struct Quad& q, const point3& Q, const vec3& u, const vec3&v) {
        q.Q = Q;
        q.u = u;
        q.v = v;
        auto n = cross(u, v);
        q.normal = unit_vector(n);
        q.D = dot(q.normal, Q);
        q.w = n / dot(n, n);
    }

    __device__ bool quad_hit(const struct Quad& q, const ray& r, interval ray_t, hit_record& rec, curandState *rand_state) const {
        auto denom = dot(q.normal, r.direction());
        //check parallel
        if(std::fabs(denom) < 1e-6) return false;

        auto t = (q.D - dot(q.normal, r.origin())) / denom;
        if(!ray_t.contains(t)) return false;
        
        //hit point
        auto intersection = r.at(t);
        //check hit in the quad
        vec3 planar_hitpt_vector = intersection - q.Q;
        auto alpha = dot(q.w, cross(planar_hitpt_vector, q.v));
        auto beta = dot(q.w, cross(q.u, planar_hitpt_vector));

        if(!is_interior(alpha, beta, rec)) return false; 

        rec.t = t;
        rec.p = intersection;
        rec.mat = mat;
        rec.set_face_normal(r, q.normal);

        return true;
    }

    __device__ bool is_interior(float a, float b, hit_record& rec) const {
        interval unit_interval = interval(0, 1);

        if(!unit_interval.contains(a) || !unit_interval.contains(b)) return false;

        rec.u = a;
        rec.v = b;
        return true;
    }
};

#endif