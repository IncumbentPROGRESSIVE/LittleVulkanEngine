#version 450

layout(location = 0) in vec2 inPos;   // Position Input
layout(location = 1) in vec3 inColor; // Color Input

layout(location = 0) out vec3 fragColor; // Pass to Fragment Shader

void main() {
    gl_Position = vec4(inPos, 0.0, 1.0); // Ensure position is passed correctly
    fragColor = inColor;  // ✅ Now actually using the color attribute
}
