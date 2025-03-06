#version 450

layout(location = 0) in vec2 inPos;   // Vertex Position
layout(location = 1) in vec3 inColor; // Vertex Color

layout(location = 0) out vec3 fragColor; // Pass color to fragment shader

void main() {
    gl_Position = vec4(inPos, 0.0, 1.0); // Use position properly
    fragColor = inColor; // Pass color to fragment shader
}
