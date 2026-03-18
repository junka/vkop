#version 450
layout(location = 0) out vec2 fragTexCoord;

// 定义覆盖全屏的坐标 (-1 to 1)
vec2 positions[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    // 将坐标映射到 0..1 纹理空间
    fragTexCoord = (positions[gl_VertexIndex] * 0.5) + 0.5;
    // 修正 Y 轴 (Vulkan 坐标系 Y 向下，纹理通常 Y 向上，视情况翻转)
    fragTexCoord.y = fragTexCoord.y; 
}