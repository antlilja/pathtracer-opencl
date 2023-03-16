cmake_minimum_required(VERSION 3.7)

project(pathtracer 
	VERSION 0.0.1 
	LANGUAGES CXX
)

add_executable(pathtracer 
	src/main.cpp
)
target_compile_features(pathtracer PRIVATE cxx_std_17)

find_package(OpenCL REQUIRED)
find_package(glm REQUIRED)
target_link_libraries(pathtracer PRIVATE OpenCL::OpenCL glm)

