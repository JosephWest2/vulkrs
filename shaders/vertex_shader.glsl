#version 460

layout(location = 0) in vec3 position;
layout(set = 0, binding = 0) uniform CameraUniform {
    mat4 view_proj;
} cameraUniform;

void main() {
    gl_Position = cameraUniform.view_proj * vec4(position, 1.0);
}
