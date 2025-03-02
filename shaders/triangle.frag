#version 450
layout(location = 0) in vec3 fragColor;  // 🛑 Input from vertex shader
layout(location = 0) out vec4 outColor;  // 🛑 Output final color

void main() {
    outColor = vec4(fragColor, 1.0);  // Apply color
}
