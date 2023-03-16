#include <cstdint>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <random>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <CL/opencl.hpp>
#include <glm/glm.hpp>

constexpr auto infinity = std::numeric_limits<float>::infinity();

struct __attribute__((packed)) object_t {
    cl_float3 min;
    cl_float3 max;
    cl_uint start;
    cl_uint end;
    cl_uint material_id;
};

struct __attribute__((packed)) material_t {
    cl_float3 albedo;
    cl_float3 emission;
    cl_float roughness;
};

struct __attribute__((packed)) camera_t {
    cl_float3 origin;
    cl_float3 horizontal;
    cl_float3 vertical;
    cl_float3 lower_left;

    static camera_t look_at(glm::vec3 origin, glm::vec3 target, float fov,
                            float aspect) {
        const auto viewport_height = 2.0f * tanf(fov * 0.5f);
        const auto viewport_width = aspect * viewport_height;

        const auto f = glm::normalize(origin - target);
        const auto h =
            glm::normalize(glm::cross(glm::vec3{0.0f, 1.0f, 0.0}, f));
        const auto v = glm::cross(f, h);

        const auto horizontal = h * viewport_width;
        const auto vertical = v * viewport_height;

        const auto lower_left =
            origin - (horizontal * 0.5f) + (vertical * 0.5f) - f;

        return camera_t{
            {origin.x, origin.y, origin.z},
            {horizontal.x, horizontal.y, horizontal.z},
            {vertical.x, vertical.y, vertical.z},
            {lower_left.x, lower_left.y, lower_left.z},
        };
    }
};

bool loadOBJ(const char* path, std::vector<cl_float3>& out_vertices,
             std::vector<uint32_t>& out_indices,
             std::vector<object_t>& out_objects,
             std::vector<material_t>& out_materials) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path))
        return false;

    std::vector<std::pair<bool, uint32_t>> unique(attrib.vertices.size(),
                                                  {false, 0});
    for (const auto& shape : shapes) {
        const auto start = out_indices.size();
        cl_float3 min = {infinity, infinity, infinity};
        cl_float3 max = {-infinity, -infinity, -infinity};
        for (const auto& index : shape.mesh.indices) {
            if (unique[index.vertex_index].first == false) {
                unique[index.vertex_index] = {
                    true,
                    static_cast<uint32_t>(out_vertices.size()),
                };
                out_vertices.emplace_back(cl_float3{
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2],
                });

                for (int i = 0; i < 3; ++i) {
                    min.s[i] = fmin(
                        min.s[i], attrib.vertices[3 * index.vertex_index + i]);
                    max.s[i] = fmax(
                        max.s[i], attrib.vertices[3 * index.vertex_index + i]);
                }
            }
            out_indices.push_back(unique[index.vertex_index].second);
        }

        out_objects.push_back({
            min,
            max,
            static_cast<cl_uint>(start),
            static_cast<cl_uint>(out_indices.size()),
            static_cast<cl_uint>(shape.mesh.material_ids[0]),
        });
    }

    for (const auto& material : materials) {
        out_materials.push_back(
            {{material.diffuse[0], material.diffuse[1], material.diffuse[2]},
             {material.emission[0], material.emission[1], material.emission[2]},
             material.roughness});
    }

    return true;
}

std::string read_file(const char* path) {
    std::ifstream stream(path);
    std::string str((std::istreambuf_iterator<char>(stream)),
                    std::istreambuf_iterator<char>());
    return str;
}

int main(int arg, char* args[]) {
    const uint32_t width = 2048;
    const uint32_t height = 2048;

    const auto camera = camera_t::look_at(
        {0.0f, 0.0f, 4.0f}, {0.0f, 0.0f, 0.0f}, 3.14f * 0.5f,
        static_cast<float>(width) / static_cast<float>(height));

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
        std::cout << "No OpenCL platforms found\n";
        return 1;
    }

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device device = devices[0];
    cl::Context context(device);

    cl::Program::Sources sources;
    std::string kernel_code = read_file("cl/main.cl");
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);
    if (program.build({device}) != CL_SUCCESS) {
        std::cout << " Error building: "
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << '\n';
        return 1;
    }

    // Load models from OBJ file
    std::vector<cl_float3> h_vertices;
    std::vector<cl_uint> h_indices;
    std::vector<object_t> h_objects;
    std::vector<material_t> h_materials;
    if (!loadOBJ(args[1], h_vertices, h_indices, h_objects, h_materials)) {
        std::cout << "Failed to load model: " << args[1] << '\n';
        return 1;
    }

    // Setup device image that we'll write to
    const cl::ImageFormat format(CL_RGBA, CL_UNORM_INT8);
    cl::Image2D image(context, CL_MEM_WRITE_ONLY, format, width, height, 0,
                      NULL);

    // Setup device buffers for render data
    cl::Buffer d_vertices =
        cl::Buffer(context, begin(h_vertices), end(h_vertices), true);
    cl::Buffer d_indices =
        cl::Buffer(context, begin(h_indices), end(h_indices), true);
    cl::Buffer d_objects =
        cl::Buffer(context, begin(h_objects), end(h_objects), true);
    cl::Buffer d_materials =
        cl::Buffer(context, begin(h_materials), end(h_materials), true);

    cl::Kernel kernel(program, "render");
    kernel.setArg(0, image);
    kernel.setArg(1, camera);
    kernel.setArg(2, d_vertices);
    kernel.setArg(3, d_indices);
    kernel.setArg(4, d_objects);
    kernel.setArg(5, d_materials);
    kernel.setArg(6, (cl_uint)(h_objects.size()));

    cl::CommandQueue queue(context, device, 0, NULL);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(width, height), cl::NDRange(32, 32));

    // Wait until kernel is finished
    queue.finish();

    // Read image data from device
    std::vector<uint8_t> h_image(width * height * 4);
    {
        cl::array<cl::size_type, 3> origin{0, 0, 0};
        cl::array<cl::size_type, 3> size{width, height, 1};
        queue.enqueueReadImage(image, CL_TRUE, origin, size, 0, 0,
                               h_image.data());
    }

    // Write image to a PPM image file
    {
        std::ofstream stream("image.ppm");
        stream << "P3\n" << width << " " << height << "\n255 \n";
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const uint32_t r = h_image[(y * width + x) * 4 + 0];
                const uint32_t g = h_image[(y * width + x) * 4 + 1];
                const uint32_t b = h_image[(y * width + x) * 4 + 2];
                stream << r << ", " << g << ", " << b << '\n';
            }
        }
    }
    return 0;
}
