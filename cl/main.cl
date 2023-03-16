struct __attribute__((packed)) camera_t {
    float3 origin;
    float3 horizontal;
    float3 vertical;
    float3 lower_left;
};

struct __attribute__((packed)) object_t {
    float3 min;
    float3 max;
    uint start;
    uint end;
    uint material_id;
};

struct __attribute__((packed)) material_t {
    float3 albedo;
    float3 emission;
    float roughness;
};

uint hash(uint x) {
    x = (x ^ 61) ^ (x >> 16);
    x = x + (x << 3);
    x = x ^ (x >> 4);
    x = x * 0x27d4eb2d;
    x = x ^ (x >> 15);
    return x;
}

float rand_float(uint x, uint y, uint bounce, uint triangle_index,
                 uint rand_num) {
    uint sample = bounce * 3 + rand_num;

    uint seed = hash(x);
    seed = hash(seed + hash(y));
    seed = hash(seed + hash(sample));
    seed = hash(seed + hash(triangle_index));
    return seed * 2.3283064365386963e-10f;
}

float3 rand_unit(uint x, uint y, uint bounce, uint triangle_index) {
    float a = rand_float(x, y, bounce, triangle_index, 0) * 2.0f * 3.14f;
    float z = (rand_float(x, y, bounce, triangle_index, 1) - 0.5f) * 2.0f;
    float r = sqrt(1.0f - z * z);
    return (float3)(r * cos(a), r * sin(a), z);
}

float triangle_intersect(float3 ray_origin, float3 ray_dir, float3 v0,
                         float3 v1, float3 v2) {
    float3 v0v1 = v1 - v0;
    float3 v0v2 = v2 - v0;
    float3 pvec = cross(ray_dir, v0v2);
    float det = dot(v0v1, pvec);

    if (det < 0.0) return -1.0f;

    float inv_det = 1.0 / det;

    float3 tvec = ray_origin - v0;
    float u = dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0) return -1.0f;

    float3 qvec = cross(tvec, v0v1);
    float v = dot(ray_dir, qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0) return -1.0f;

    return dot(v0v2, qvec) * inv_det;
}

bool aabb_intersect(float3 ray_origin, float3 ray_dir, float3 min, float3 max) {
    float3 inv_d = (float3)(1.0, 1.0, 1.0) / ray_dir;
    float3 t0 = (min - ray_origin) * inv_d;
    float3 t1 = (max - ray_origin) * inv_d;
    for (int i = 0; i < 3; ++i) {
        if (t0[i] < 0.0f && t1[i] < 0.0f) return true;
    }
    return false;
}

void kernel render(__write_only image2d_t image, struct camera_t camera,
                   __constant float3* vertices, __constant uint* indices,
                   __constant struct object_t* objects,
                   __constant struct material_t* materials, uint num_objects) {
    int width = get_global_size(0);
    int height = get_global_size(1);
    int x = get_global_id(0);
    int y = get_global_id(1);

    float u = ((float)x) / ((float)width);
    float v = ((float)y) / ((float)height);

    float3 ray_origin = camera.origin;
    float3 ray_dir = normalize(camera.lower_left + camera.horizontal * u -
                               camera.vertical * v - ray_origin);

    float3 color = (float3)(0.0, 0.0, 0.0);
    float3 atten = (float3)(1.0, 1.0, 1.0);
    for (int b = 0; b < 4; ++b) {
        float min_t = 100000.0;
        uint material_id = 10000;
        uint index;

        for (uint o = 0; o < num_objects; ++o) {
            if (aabb_intersect(ray_origin, ray_dir, objects[o].min,
                               objects[o].max))
                continue;

            for (uint i = objects[o].start; i < objects[o].end; i += 3) {
                float3 v0 = vertices[indices[i + 0]];
                float3 v1 = vertices[indices[i + 1]];
                float3 v2 = vertices[indices[i + 2]];

                float t = triangle_intersect(ray_origin, ray_dir, v0, v1, v2);

                if (t >= 0.0f && t < min_t) {
                    min_t = t;
                    material_id = objects[o].material_id;
                    index = i / 3;
                }
            }
        }

        if (material_id != 10000) {
            float3 p = ray_origin + min_t * ray_dir;

            float3 v0 = vertices[indices[index * 3 + 0]];
            float3 v1 = vertices[indices[index * 3 + 1]];
            float3 v2 = vertices[indices[index * 3 + 2]];

            struct material_t material = materials[material_id];

            float3 normal = normalize(cross(v1 - v0, v2 - v0));

            float3 diffuse = normal + rand_unit(x, y, b, index);
            float3 reflect = ray_dir - (normal * (dot(normal, ray_dir) * 2.0f));
            ray_dir = normalize((1.0f - material.roughness) * reflect +
                                material.roughness * diffuse);

            ray_origin = p + ray_dir * 0.0001f;

            color += atten * material.emission;
            atten *= material.albedo * 0.5f;
        }
        else {
            float t = 0.5f * (ray_dir.y + 1.0f);
            float3 white = (float3)(1.0, 1.0, 1.0);
            float3 blue = (float3)(0.5, 0.7, 1.0);
            float3 sky = (1.0f - t) * white + t * blue;
            color += atten * sky;
            break;
        }
    }

    color =
        clamp(sqrt(color), (float3)(0.0, 0.0, 0.0), (float3)(1.0, 1.0, 1.0));

    write_imagef(image, (int2)(x, y), (float4)(color, 1.0));
}
