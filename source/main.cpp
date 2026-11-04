#include <gx2/draw.h>
#include <gx2/state.h>
#include <gx2/registers.h>
#include <gx2r/buffer.h>
#include <gx2r/draw.h>
#include <vpad/input.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "crema_app.h"
#include "crema_buffer.h"
#include "crema_frame.h"
#include "crema_matrix.h"
#include "crema_shader.h"

// ============================================================
// THE PARK - first-person Wii U park
// ============================================================

static const float PI = 3.14159265358979323846f;

static float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static float wrapAngle(float a)
{
    while (a > PI) a -= 2.0f * PI;
    while (a < -PI) a += 2.0f * PI;
    return a;
}

static float angleDifference(float a, float b)
{
    return wrapAngle(a - b);
}

// ============================================================
// SHADERS
// ============================================================

static const char *VERTEX_SHADER =
    "#version 450\n"
    "layout(location=0) in vec3 aPosition;\n"
    "layout(location=1) in vec3 aNormal;\n"
    "layout(location=2) in vec3 aColor;\n"

    "layout(binding=0) uniform Global {\n"
    " mat4 uViewProj;\n"
    " vec4 uLightDir;\n"
    " vec4 uCamPos;\n"
    " vec4 uFogParam;\n"
    " vec4 uFogColor;\n"
    "};\n"

    "layout(location=0) out vec3 vColor;\n"
    "layout(location=1) out vec3 vWorld;\n"
    "layout(location=2) out vec3 vNormal;\n"

    "void main() {\n"
    " gl_Position = uViewProj * vec4(aPosition,1.0);\n"
    " vColor = aColor;\n"
    " vWorld = aPosition;\n"
    " vNormal = aNormal;\n"
    "}\n";

static const char *PIXEL_SHADER =
    "#version 450\n"
    "layout(location=0) in vec3 vColor;\n"
    "layout(location=1) in vec3 vWorld;\n"
    "layout(location=2) in vec3 vNormal;\n"

    "layout(binding=0) uniform Global {\n"
    " mat4 uViewProj;\n"
    " vec4 uLightDir;\n"
    " vec4 uCamPos;\n"
    " vec4 uFogParam;\n"
    " vec4 uFogColor;\n"
    "};\n"

    "layout(location=0) out vec4 oColor;\n"

    "void main() {\n"
    " vec3 n = normalize(vNormal);\n"
    " float diffuse = max(dot(n,-uLightDir.xyz),0.0);\n"
    " float light = 0.30 + diffuse * 0.70;\n"
    " vec3 color = vColor * light;\n"
    " float distanceFromCamera = length(vWorld-uCamPos.xyz);\n"
    " float fog = smoothstep(uFogParam.x,uFogParam.y,distanceFromCamera);\n"
    " color = mix(color,uFogColor.rgb,fog);\n"
    " oColor = vec4(color,1.0);\n"
    "}\n";

// ============================================================
// PROCEDURAL PARK GEOMETRY
// ============================================================

struct Vertex
{
    float position[3];
    float normal[3];
    float color[3];
};

static std::vector<Vertex> vertices;
static std::vector<uint16_t> indices;

static void addTriangle(
    const float *a,
    const float *b,
    const float *c,
    const float *color)
{
    if (vertices.size() > 65530)
        return;

    float ux = b[0] - a[0];
    float uy = b[1] - a[1];
    float uz = b[2] - a[2];

    float vx = c[0] - a[0];
    float vy = c[1] - a[1];
    float vz = c[2] - a[2];

    float nx = uy * vz - uz * vy;
    float ny = uz * vx - ux * vz;
    float nz = ux * vy - uy * vx;

    float length = sqrtf(nx * nx + ny * ny + nz * nz);
    if (length > 0.00001f)
    {
        nx /= length;
        ny /= length;
        nz /= length;
    }

    const float *points[3] = { a, b, c };

    for (int i = 0; i < 3; ++i)
    {
        Vertex v{};

        for (int j = 0; j < 3; ++j)
        {
            v.position[j] = points[i][j];
            v.color[j] = color[j];
        }

        v.normal[0] = nx;
        v.normal[1] = ny;
        v.normal[2] = nz;

        indices.push_back((uint16_t)vertices.size());
        vertices.push_back(v);
    }
}

static void addQuad(
    const float *a,
    const float *b,
    const float *c,
    const float *d,
    const float *color)
{
    addTriangle(a, b, c, color);
    addTriangle(a, c, d, color);
}

static void addBox(
    float x, float y, float z,
    float sx, float sy, float sz,
    const float *color)
{
    float x0 = x - sx * 0.5f;
    float x1 = x + sx * 0.5f;
    float y0 = y;
    float y1 = y + sy;
    float z0 = z - sz * 0.5f;
    float z1 = z + sz * 0.5f;

    float p[8][3] =
    {
        {x0,y0,z0}, {x1,y0,z0},
        {x1,y1,z0}, {x0,y1,z0},
        {x0,y0,z1}, {x1,y0,z1},
        {x1,y1,z1}, {x0,y1,z1}
    };

    addQuad(p[0],p[3],p[2],p[1],color);
    addQuad(p[4],p[5],p[6],p[7],color);
    addQuad(p[0],p[4],p[7],p[3],color);
    addQuad(p[1],p[2],p[6],p[5],color);
    addQuad(p[0],p[1],p[5],p[4],color);
    addQuad(p[3],p[7],p[6],p[2],color);
}

static void addCylinder(
    float x, float y, float z,
    float radius, float height,
    const float *color)
{
    const int segments = 10;

    for (int i = 0; i < segments; ++i)
    {
        float a = (float)i / segments * 2.0f * PI;
        float b = (float)(i + 1) / segments * 2.0f * PI;

        float p0[3] =
        {
            x + cosf(a) * radius, y, z + sinf(a) * radius
        };

        float p1[3] =
        {
            x + cosf(b) * radius, y, z + sinf(b) * radius
        };

        float p2[3] =
        {
            p1[0], y + height, p1[2]
        };

        float p3[3] =
        {
            p0[0], y + height, p0[2]
        };

        addQuad(p0,p3,p2,p1,color);
    }
}

static void addCone(
    float x, float y, float z,
    float radius, float height,
    const float *color)
{
    const int segments = 12;

    float top[3] = { x, y + height, z };

    for (int i = 0; i < segments; ++i)
    {
        float a = (float)i / segments * 2.0f * PI;
        float b = (float)(i + 1) / segments * 2.0f * PI;

        float p0[3] =
        {
            x + cosf(a) * radius, y, z + sinf(a) * radius
        };

        float p1[3] =
        {
            x + cosf(b) * radius, y, z + sinf(b) * radius
        };

        addTriangle(p0,top,p1,color);
    }
}

static void addEllipsoid(
    float x, float y, float z,
    float rx, float ry, float rz,
    const float *color)
{
    const int rings = 7;
    const int sides = 12;

    for (int r = 0; r < rings; ++r)
    {
        float lat0 = -PI * 0.5f + PI * r / rings;
        float lat1 = -PI * 0.5f + PI * (r + 1) / rings;

        for (int s = 0; s < sides; ++s)
        {
            float a0 = 2.0f * PI * s / sides;
            float a1 = 2.0f * PI * (s + 1) / sides;

            float p0[3] =
            {
                x + rx*cosf(lat0)*cosf(a0),
                y + ry*sinf(lat0),
                z + rz*cosf(lat0)*sinf(a0)
            };

            float p1[3] =
            {
                x + rx*cosf(lat0)*cosf(a1),
                y + ry*sinf(lat0),
                z + rz*cosf(lat0)*sinf(a1)
            };

            float p2[3] =
            {
                x + rx*cosf(lat1)*cosf(a1),
                y + ry*sinf(lat1),
                z + rz*cosf(lat1)*sinf(a1)
            };

            float p3[3] =
            {
                x + rx*cosf(lat1)*cosf(a0),
                y + ry*sinf(lat1),
                z + rz*cosf(lat1)*sinf(a0)
            };

            addQuad(p0,p1,p2,p3,color);
        }
    }
}

// ============================================================
// PARK OBJECTS
// ============================================================

static const float GRASS[]   = {0.20f,0.42f,0.16f};
static const float DARKGRASS[] = {0.14f,0.32f,0.12f};
static const float PATH[]    = {0.53f,0.47f,0.36f};
static const float WOOD[]    = {0.35f,0.18f,0.08f};
static const float WOOD2[]   = {0.53f,0.30f,0.13f};
static const float LEAVES[]  = {0.10f,0.32f,0.10f};
static const float LEAVES2[] = {0.18f,0.42f,0.13f};
static const float WATER[]   = {0.08f,0.30f,0.48f};
static const float METAL[]   = {0.20f,0.22f,0.23f};
static const float LAMP[]    = {0.90f,0.78f,0.48f};
static const float FLOWER[]  = {0.72f,0.30f,0.35f};

static void buildGround()
{
    float a[3] = {-42.0f,0.0f,-42.0f};
    float b[3] = {-42.0f,0.0f, 42.0f};
    float c[3] = { 42.0f,0.0f, 42.0f};
    float d[3] = { 42.0f,0.0f,-42.0f};

    addQuad(a,b,c,d,GRASS);

    // Outer park boundary.
    addBox(0.0f,-0.3f,-42.0f,84.0f,0.6f,0.5f,DARKGRASS);
    addBox(0.0f,-0.3f, 42.0f,84.0f,0.6f,0.5f,DARKGRASS);
    addBox(-42.0f,-0.3f,0.0f,0.5f,0.6f,84.0f,DARKGRASS);
    addBox( 42.0f,-0.3f,0.0f,0.5f,0.6f,84.0f,DARKGRASS);
}

static void buildPath()
{
    // Winding central walking path.
    const float points[][2] =
    {
        {0.0f,35.0f},
        {0.0f,25.0f},
        {3.0f,17.0f},
        {5.0f,9.0f},
        {3.0f,2.0f},
        {0.0f,-5.0f},
        {-4.0f,-12.0f},
        {-5.0f,-21.0f},
        {-2.0f,-32.0f}
    };

    for (int i = 0; i < 8; ++i)
    {
        float x0 = points[i][0];
        float z0 = points[i][1];
        float x1 = points[i+1][0];
        float z1 = points[i+1][1];

        float dx = x1-x0;
        float dz = z1-z0;
        float len = sqrtf(dx*dx+dz*dz);

        if (len < 0.001f) continue;

        float px = -dz/len * 2.0f;
        float pz = dx/len * 2.0f;

        float a[3] = {x0+px,0.025f,z0+pz};
        float b[3] = {x1+px,0.025f,z1+pz};
        float c[3] = {x1-px,0.025f,z1-pz};
        float d[3] = {x0-px,0.025f,z0-pz};

        addQuad(a,b,c,d,PATH);
    }
}

static void buildPond()
{
    const float cx = 12.0f;
    const float cz = -8.0f;
    const float rx = 7.0f;
    const float rz = 5.0f;

    float center[3] = {cx,0.06f,cz};

    for (int i = 0; i < 32; ++i)
    {
        float a = 2.0f*PI*i/32.0f;
        float b = 2.0f*PI*(i+1)/32.0f;

        float p0[3] =
        {
            cx+cosf(a)*rx,0.06f,cz+sinf(a)*rz
        };

        float p1[3] =
        {
            cx+cosf(b)*rx,0.06f,cz+sinf(b)*rz
        };

        addTriangle(center,p1,p0,WATER);
    }

    // Bridge deck across the pond.
    addBox(12.0f,0.35f,-8.0f,17.0f,0.35f,2.4f,WOOD2);

    // Bridge railings.
    addBox(12.0f,1.15f,-9.2f,17.0f,1.2f,0.15f,WOOD);
    addBox(12.0f,1.15f,-6.8f,17.0f,1.2f,0.15f,WOOD);

    // Supporting bridge posts.
    for (int i = 0; i < 7; ++i)
    {
        float x = 4.5f + i*2.5f;
        addBox(x,0.35f,-9.2f,0.18f,1.1f,0.18f,WOOD2);
        addBox(x,0.35f,-6.8f,0.18f,1.1f,0.18f,WOOD2);
    }
}

static void buildTree(float x, float z, float size)
{
    addCylinder(x,0.0f,z,0.25f*size,2.5f*size,WOOD);

    addCone(x,1.6f*size,z,1.5f*size,2.8f*size,LEAVES);
    addCone(x,2.6f*size,z,1.15f*size,2.5f*size,LEAVES2);
    addCone(x,3.5f*size,z,0.75f*size,1.9f*size,LEAVES);
}

static void buildBench(float x, float z)
{
    // Seat.
    addBox(x,0.75f,z,2.2f,0.15f,0.65f,WOOD2);

    // Backrest.
    addBox(x,1.25f,z+0.28f,2.2f,0.85f,0.13f,WOOD);

    // Legs.
    addBox(x-0.85f,0.0f,z,0.13f,0.75f,0.13f,METAL);
    addBox(x+0.85f,0.0f,z,0.13f,0.75f,0.13f,METAL);
}

static void buildLamp(float x, float z)
{
    addCylinder(x,0.0f,z,0.10f,4.0f,METAL);
    addBox(x,4.0f,z,0.75f,0.22f,0.75f,LAMP);
}

static void buildPark()
{
    buildGround();
    buildPath();
    buildPond();

    // Trees around the park.
    buildTree(-27.0f, 27.0f,1.2f);
    buildTree(-18.0f, 31.0f,0.9f);
    buildTree(-31.0f, 12.0f,1.1f);
    buildTree(-27.0f,-3.0f,1.0f);
    buildTree(-30.0f,-24.0f,1.2f);
    buildTree(-17.0f,-31.0f,0.9f);
    buildTree( 28.0f, 30.0f,1.1f);
    buildTree( 34.0f, 16.0f,1.0f);
    buildTree( 31.0f,  1.0f,1.2f);
    buildTree( 29.0f,-24.0f,1.0f);
    buildTree( 19.0f,-32.0f,1.1f);
    buildTree(-15.0f,  4.0f,0.9f);
    buildTree(-19.0f,-12.0f,1.0f);
    buildTree( 23.0f, 18.0f,0.9f);

    // Bushes.
    addEllipsoid(-11.0f,0.65f,22.0f,1.5f,0.8f,1.2f,LEAVES2);
    addEllipsoid(-13.0f,0.55f,20.0f,1.2f,0.65f,1.0f,LEAVES);
    addEllipsoid( 20.0f,0.65f,23.0f,1.5f,0.8f,1.2f,LEAVES2);
    addEllipsoid( 22.0f,0.55f,21.0f,1.2f,0.65f,1.0f,LEAVES);

    // Benches.
    buildBench(-9.0f,14.0f);
    buildBench( 0.0f,-25.0f);
    buildBench(23.0f,  4.0f);

    // Lamp posts.
    buildLamp(-7.0f,22.0f);
    buildLamp( 8.0f,13.0f);
    buildLamp( 7.0f,-1.0f);
    buildLamp(-8.0f,-17.0f);
    buildLamp( 9.0f,-27.0f);

    // Flower patches.
    for (int i = 0; i < 12; ++i)
    {
        float x = -22.0f + (float)(i%4)*1.0f;
        float z =  8.0f + (float)(i/4)*1.0f;
        addEllipsoid(x,0.18f,z,0.18f,0.15f,0.18f,FLOWER);
    }
}

// ============================================================
// RENDERING
// ============================================================

struct GlobalBlock
{
    Mat4 viewProj;
    float lightDir[4];
    float camPos[4];
    float fogParam[4];
    float fogColor[4];
};

struct SceneView
{
    const CremaShader *shader;
    GX2RBuffer *vertexBuffer;
    GX2RBuffer *indexBuffer;
    const void *uniforms;
    int32_t vertexBlock;
    int32_t pixelBlock;
    uint32_t indexCount;
};

static void drawPark(void *user)
{
    SceneView *scene = (SceneView *)user;

    GX2SetDepthOnlyControl(
        TRUE, TRUE, GX2_COMPARE_FUNC_LESS);

    // We use face normals and disable face culling so that
    // every generated park surface remains visible.
    GX2SetCullOnlyControl(
        GX2_FRONT_FACE_CCW, FALSE, FALSE);

    CremaShaderBind(scene->shader);

    GX2SetVertexUniformBlock(
        scene->vertexBlock,
        sizeof(GlobalBlock),
        scene->uniforms);

    GX2SetPixelUniformBlock(
        scene->pixelBlock,
        sizeof(GlobalBlock),
        scene->uniforms);

    GX2RSetAttributeBuffer(
        scene->vertexBuffer,
        0,
        sizeof(Vertex),
        0);

    GX2RDrawIndexed(
        GX2_PRIMITIVE_MODE_TRIANGLES,
        scene->indexBuffer,
        GX2_INDEX_TYPE_U16,
        scene->indexCount,
        0, 0, 1);
}

// ============================================================
// GAMEPAD ORIENTATION
// ============================================================

static float baseYaw = 0.0f;
static float basePitch = 0.0f;
static float cameraYaw = 0.0f;
static float cameraPitch = 0.0f;

static bool readGamePad(VPADStatus *status)
{
    VPADReadError error;
    return VPADRead(
        VPAD_CHAN_0,
        status,
        1,
        &error) > 0;
}

static bool getPadAngles(
    const VPADStatus &pad,
    float *yaw,
    float *pitch)
{
    float fx = pad.direction.z.x;
    float fy = pad.direction.z.y;
    float fz = pad.direction.z.z;

    float length = sqrtf(fx*fx + fy*fy + fz*fz);

    if (length < 0.5f)
        return false;

    fx /= length;
    fy /= length;
    fz /= length;

    *yaw = atan2f(-fx,-fz);
    *pitch = asinf(clampf(fy,-1.0f,1.0f));

    return true;
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char **argv)
{
    if (!CremaAppInit("The Park"))
        return -1;

    if (!CremaShaderInitCompiler())
    {
        CremaAppShutdown();
        return -1;
    }

    // Generate the entire park.
    buildPark();

    if (vertices.empty() || indices.empty())
    {
        CremaShaderShutdownCompiler();
        CremaAppShutdown();
        return -1;
    }

    const CremaAttrib attributes[] =
    {
        {0,0,offsetof(Vertex,position),
            GX2_ATTRIB_FORMAT_FLOAT_32_32_32},

        {1,0,offsetof(Vertex,normal),
            GX2_ATTRIB_FORMAT_FLOAT_32_32_32},

        {2,0,offsetof(Vertex,color),
            GX2_ATTRIB_FORMAT_FLOAT_32_32_32}
    };

    CremaShader *shader = CremaShaderCompile(
        VERTEX_SHADER,
        PIXEL_SHADER,
        attributes,
        3);

    if (!shader)
    {
        CremaShaderShutdownCompiler();
        CremaAppShutdown();
        return -1;
    }

    int32_t vertexBlock =
        CremaShaderVSBlockLocation(shader,"Global");

    int32_t pixelBlock =
        CremaShaderPSBlockLocation(shader,"Global");

    if (vertexBlock < 0) vertexBlock = 0;
    if (pixelBlock < 0) pixelBlock = 0;

    GX2RBuffer vertexBuffer{};
    GX2RBuffer indexBuffer{};

    bool buffersReady =
        CremaBufferCreateVertex(
            &vertexBuffer,
            sizeof(Vertex),
            (uint32_t)vertices.size(),
            vertices.data())
        &&
        CremaBufferCreateIndexU16(
            &indexBuffer,
            (uint32_t)indices.size(),
            indices.data());

    if (!buffersReady)
    {
        CremaBufferDestroy(&vertexBuffer);
        CremaBufferDestroy(&indexBuffer);
        CremaShaderFree(shader);
        CremaShaderShutdownCompiler();
        CremaAppShutdown();
        return -1;
    }

    // Per-frame uniforms are safely double-buffered.
    CremaUniformRing globals{};

    if (!CremaUniformRingCreate(
        &globals,
        sizeof(GlobalBlock),
        CREMA_FRAMES_IN_FLIGHT))
    {
        CremaBufferDestroy(&indexBuffer);
        CremaBufferDestroy(&vertexBuffer);
        CremaShaderFree(shader);
        CremaShaderShutdownCompiler();
        CremaAppShutdown();
        return -1;
    }

    CremaFrame frame;
    CremaFrameInit(&frame,CREMA_PACING_FENCED,1);

    CremaClock clock;
    CremaClockInit(&clock);

    CremaFrameStats stats{};

    // Player starts near the park entrance.
    Vec3 camPos = {0.0f,1.7f,32.0f};

    Mat4 projection = mat4_perspective(
        60.0f * PI / 180.0f,
        16.0f / 9.0f,
        0.1f,
        150.0f);

    Vec3 lightRaw = {-0.45f,-0.8f,-0.35f};
    Vec3 light = vec3_normalize(lightRaw);

    // Initial orientation calibration.
    // Keep the GamePad still in your normal playing position.
    float yawSum = 0.0f;
    float pitchSum = 0.0f;
    int calibrationSamples = 0;

    SceneView scene{};
    scene.shader = shader;
    scene.vertexBuffer = &vertexBuffer;
    scene.indexBuffer = &indexBuffer;
    scene.vertexBlock = vertexBlock;
    scene.pixelBlock = pixelBlock;
    scene.indexCount = (uint32_t)indices.size();

    while (CremaAppRunning())
    {
        CremaClockTick(&clock);
        float dt = clock.dt;

        VPADStatus pad{};

        if (readGamePad(&pad))
        {
            float measuredYaw;
            float measuredPitch;

            if (getPadAngles(
                pad,
                &measuredYaw,
                &measuredPitch))
            {
                if (calibrationSamples < 90)
                {
                    yawSum += measuredYaw;
                    pitchSum += measuredPitch;
                    calibrationSamples++;

                    if (calibrationSamples == 90)
                    {
                        baseYaw = yawSum / 90.0f;
                        basePitch = pitchSum / 90.0f;
                    }
                }
                else
                {
                    float targetYaw =
                        angleDifference(measuredYaw,baseYaw);

                    float targetPitch =
                        measuredPitch-basePitch;

                    targetPitch =
                        clampf(targetPitch,-1.2f,1.2f);

                    // Smooth the camera response.
                    float smoothing =
                        clampf(dt*12.0f,0.0f,1.0f);

                    cameraYaw +=
                        angleDifference(targetYaw,cameraYaw)
                        * smoothing;

                    cameraPitch +=
                        (targetPitch-cameraPitch)*smoothing;

                    // PLUS recenters the current GamePad orientation.
                    if (pad.trigger & VPAD_BUTTON_PLUS)
                    {
                        baseYaw = measuredYaw;
                        basePitch = measuredPitch;
                        cameraYaw = 0.0f;
                        cameraPitch = 0.0f;
                    }
                }
            }

            // Left stick movement is independent of camera pitch.
            float lx = pad.leftStick.x;
            float ly = pad.leftStick.y;

            if (fabsf(lx) < 0.12f) lx = 0.0f;
            if (fabsf(ly) < 0.12f) ly = 0.0f;

            float forwardX = -sinf(cameraYaw);
            float forwardZ = -cosf(cameraYaw);

            float rightX = cosf(cameraYaw);
            float rightZ = -sinf(cameraYaw);

            const float walkingSpeed = 4.5f;

            camPos.x +=
                (forwardX*ly + rightX*lx)
                * walkingSpeed * dt;

            camPos.z +=
                (forwardZ*ly + rightZ*lx)
                * walkingSpeed * dt;
        }

        // Keep the player inside the park.
        camPos.x = clampf(camPos.x,-39.0f,39.0f);
        camPos.z = clampf(camPos.z,-39.0f,39.0f);

        uint32_t slot = CremaFrameBegin(&frame);

        Mat4 view = mat4_mul(
            mat4_rotate_x(-cameraPitch),
            mat4_mul(
                mat4_rotate_y(-cameraYaw),
                mat4_translate(
                    -camPos.x,
                    -camPos.y,
                    -camPos.z)));

        GlobalBlock block{};

        block.viewProj = mat4_mul(projection,view);

        block.lightDir[0] = light.x;
        block.lightDir[1] = light.y;
        block.lightDir[2] = light.z;
        block.lightDir[3] = 0.0f;

        block.camPos[0] = camPos.x;
        block.camPos[1] = camPos.y;
        block.camPos[2] = camPos.z;
        block.camPos[3] = 0.0f;

        block.fogParam[0] = 65.0f;
        block.fogParam[1] = 110.0f;

        block.fogColor[0] = 0.48f;
        block.fogColor[1] = 0.68f;
        block.fogColor[2] = 0.82f;
        block.fogColor[3] = 1.0f;

        scene.uniforms = CremaUniformRingStore(
            &globals,
            slot,
            &block,
            sizeof(block));

        const float sky[4] =
        {
            0.48f,0.68f,0.82f,1.0f
        };

        CremaFrameDrawBoth(
            sky,
            drawPark,
            &scene);

        CremaFrameEnd(&frame,&stats);
    }

    // Clean shutdown.
    CremaFrameSettle(&frame);

    CremaUniformRingDestroy(&globals);

    CremaBufferDestroy(&indexBuffer);
    CremaBufferDestroy(&vertexBuffer);

    CremaShaderFree(shader);
    CremaShaderShutdownCompiler();

    CremaAppShutdown();

    return 0;
}
