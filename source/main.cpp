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
// THE PARK - FOUR-HOUSE NEIGHBORHOOD GRAPHICS TEST
// Complete replacement for source/main.cpp
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
    " float light = 0.34 + diffuse * 0.66;\n"
    " vec3 color = vColor * light;\n"
    " float d = length(vWorld-uCamPos.xyz);\n"
    " float fog = smoothstep(uFogParam.x,uFogParam.y,d);\n"
    " color = mix(color,uFogColor.rgb,fog);\n"
    " oColor = vec4(color,1.0);\n"
    "}\n";

// ============================================================
// GEOMETRY STORAGE AND PRIMITIVES
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
    const float *a, const float *b, const float *c,
    const float *color)
{
    if (vertices.size() > 65530)
        return;

    float ux = b[0]-a[0], uy = b[1]-a[1], uz = b[2]-a[2];
    float vx = c[0]-a[0], vy = c[1]-a[1], vz = c[2]-a[2];

    float nx = uy*vz-uz*vy;
    float ny = uz*vx-ux*vz;
    float nz = ux*vy-uy*vx;
    float len = sqrtf(nx*nx+ny*ny+nz*nz);

    if (len > 0.00001f)
    {
        nx /= len;
        ny /= len;
        nz /= len;
    }

    const float *p[3] = {a,b,c};

    for (int i=0; i<3; ++i)
    {
        Vertex v{};
        for (int j=0; j<3; ++j)
        {
            v.position[j] = p[i][j];
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
    const float *a, const float *b,
    const float *c, const float *d,
    const float *color)
{
    addTriangle(a,b,c,color);
    addTriangle(a,c,d,color);
}

static void addBox(
    float x, float y, float z,
    float sx, float sy, float sz,
    const float *color)
{
    float x0=x-sx*0.5f, x1=x+sx*0.5f;
    float y0=y, y1=y+sy;
    float z0=z-sz*0.5f, z1=z+sz*0.5f;

    float p[8][3] =
    {
        {x0,y0,z0},{x1,y0,z0},{x1,y1,z0},{x0,y1,z0},
        {x0,y0,z1},{x1,y0,z1},{x1,y1,z1},{x0,y1,z1}
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
    const int segments=12;
    for (int i=0; i<segments; ++i)
    {
        float a=2.0f*PI*i/segments;
        float b=2.0f*PI*(i+1)/segments;
        float p0[3]={x+cosf(a)*radius,y,z+sinf(a)*radius};
        float p1[3]={x+cosf(b)*radius,y,z+sinf(b)*radius};
        float p2[3]={p1[0],y+height,p1[2]};
        float p3[3]={p0[0],y+height,p0[2]};
        addQuad(p0,p3,p2,p1,color);
    }
}

static void addCone(
    float x, float y, float z,
    float radius, float height,
    const float *color)
{
    const int segments=12;
    float top[3]={x,y+height,z};
    for (int i=0; i<segments; ++i)
    {
        float a=2.0f*PI*i/segments;
        float b=2.0f*PI*(i+1)/segments;
        float p0[3]={x+cosf(a)*radius,y,z+sinf(a)*radius};
        float p1[3]={x+cosf(b)*radius,y,z+sinf(b)*radius};
        addTriangle(p0,top,p1,color);
    }
}

static void addEllipsoid(
    float x, float y, float z,
    float rx, float ry, float rz,
    const float *color)
{
    const int rings=7, sides=12;
    for (int r=0; r<rings; ++r)
    {
        float lat0=-PI*0.5f+PI*r/rings;
        float lat1=-PI*0.5f+PI*(r+1)/rings;
        for (int s=0; s<sides; ++s)
        {
            float a0=2.0f*PI*s/sides;
            float a1=2.0f*PI*(s+1)/sides;
            float p0[3]={x+rx*cosf(lat0)*cosf(a0),y+ry*sinf(lat0),z+rz*cosf(lat0)*sinf(a0)};
            float p1[3]={x+rx*cosf(lat0)*cosf(a1),y+ry*sinf(lat0),z+rz*cosf(lat0)*sinf(a1)};
            float p2[3]={x+rx*cosf(lat1)*cosf(a1),y+ry*sinf(lat1),z+rz*cosf(lat1)*sinf(a1)};
            float p3[3]={x+rx*cosf(lat1)*cosf(a0),y+ry*sinf(lat1),z+rz*cosf(lat1)*sinf(a0)};
            addQuad(p0,p1,p2,p3,color);
        }
    }
}

// ============================================================
// MATERIAL COLORS
// ============================================================

static const float GRASS[]       ={0.18f,0.37f,0.15f};
static const float GRASS_LIGHT[] ={0.23f,0.43f,0.18f};
static const float GRASS_DARK[]  ={0.13f,0.29f,0.11f};
static const float ASPHALT[]     ={0.14f,0.15f,0.16f};
static const float ROAD_EDGE[]   ={0.23f,0.24f,0.24f};
static const float ROAD_LINE[]   ={0.83f,0.79f,0.66f};
static const float CONCRETE[]    ={0.53f,0.54f,0.52f};
static const float CONCRETE2[]   ={0.66f,0.65f,0.60f};
static const float BRICK[]       ={0.55f,0.25f,0.17f};
static const float BRICK_DARK[]  ={0.35f,0.16f,0.12f};
static const float SIDING[]      ={0.76f,0.71f,0.61f};
static const float BLUE_SIDING[] ={0.32f,0.44f,0.54f};
static const float GREY_SIDING[] ={0.48f,0.49f,0.47f};
static const float WHITE[]       ={0.88f,0.86f,0.79f};
static const float TRIM[]        ={0.95f,0.92f,0.83f};
static const float ROOF[]        ={0.15f,0.17f,0.20f};
static const float ROOF_BROWN[]  ={0.29f,0.22f,0.17f};
static const float WOOD[]        ={0.31f,0.18f,0.10f};
static const float WOOD_LIGHT[]  ={0.52f,0.33f,0.19f};
static const float GLASS[]       ={0.18f,0.37f,0.47f};
static const float GLASS_LIGHT[] ={0.39f,0.58f,0.66f};
static const float METAL[]       ={0.18f,0.20f,0.21f};
static const float BLACK[]       ={0.06f,0.07f,0.08f};
static const float LEAVES[]      ={0.10f,0.29f,0.11f};
static const float LEAVES2[]     ={0.19f,0.40f,0.14f};
static const float FLOWER[]      ={0.78f,0.32f,0.37f};
static const float LAMP_COLOR[]  ={0.93f,0.82f,0.57f};

// ============================================================
// ROAD, SIDEWALKS AND YARDS
// ============================================================

static void buildGround()
{
    addBox(0.0f,-0.35f,0.0f,84.0f,0.65f,84.0f,GRASS);

    for (int i=0; i<12; ++i)
    {
        float z=-32.0f+i*5.8f;
        addBox(-14.0f,0.005f,z,8.0f,0.025f,5.2f,GRASS_LIGHT);
        addBox( 14.0f,0.005f,z,8.0f,0.025f,5.2f,GRASS_DARK);
    }

    // Low outer boundary keeps the test area visually contained.
    addBox(0.0f,-0.05f,-41.5f,84.0f,0.35f,0.35f,GRASS_DARK);
    addBox(0.0f,-0.05f, 41.5f,84.0f,0.35f,0.35f,GRASS_DARK);
    addBox(-41.5f,-0.05f,0.0f,0.35f,0.35f,84.0f,GRASS_DARK);
    addBox( 41.5f,-0.05f,0.0f,0.35f,0.35f,84.0f,GRASS_DARK);
}

static void buildRoad()
{
    addBox(0.0f,-0.035f,0.0f,13.5f,0.07f,76.0f,ASPHALT);

    addBox(-6.85f,0.0f,0.0f,0.25f,0.09f,76.0f,ROAD_EDGE);
    addBox( 6.85f,0.0f,0.0f,0.25f,0.09f,76.0f,ROAD_EDGE);

    // Broken center line.
    for (int i=0; i<18; ++i)
    {
        float z=-34.0f+i*3.9f;
        addBox(0.0f,0.012f,z,0.13f,0.025f,1.9f,ROAD_LINE);
    }

    // Sidewalks and raised curbs.
    addBox(-8.0f,0.025f,0.0f,2.0f,0.12f,76.0f,CONCRETE);
    addBox( 8.0f,0.025f,0.0f,2.0f,0.12f,76.0f,CONCRETE);
    addBox(-6.95f,0.08f,0.0f,0.22f,0.22f,76.0f,CONCRETE2);
    addBox( 6.95f,0.08f,0.0f,0.22f,0.22f,76.0f,CONCRETE2);

    // Expansion joints in the sidewalks.
    for (int i=0; i<19; ++i)
    {
        float z=-35.0f+i*3.8f;
        addBox(-8.0f,0.09f,z,1.95f,0.012f,0.035f,ROAD_EDGE);
        addBox( 8.0f,0.09f,z,1.95f,0.012f,0.035f,ROAD_EDGE);
    }
}

static void buildDriveway(float x, float z)
{
    addBox(x,0.04f,z,3.0f,0.10f,7.5f,CONCRETE);
    addBox(x,0.095f,z,2.82f,0.018f,7.3f,CONCRETE2);
    addBox(x-1.47f,0.11f,z,0.06f,0.025f,7.3f,WHITE);
    addBox(x+1.47f,0.11f,z,0.06f,0.025f,7.3f,WHITE);
}

// ============================================================
// STREET PROPS
// ============================================================

static void buildTree(float x, float z, float s)
{
    addCylinder(x,0.0f,z,0.23f*s,2.5f*s,WOOD);
    addEllipsoid(x,2.55f*s,z,1.35f*s,1.35f*s,1.25f*s,LEAVES);
    addEllipsoid(x-0.55f*s,2.8f*s,z+0.25f*s,0.85f*s,0.9f*s,0.85f*s,LEAVES2);
    addEllipsoid(x+0.60f*s,2.75f*s,z-0.2f*s,0.9f*s,0.95f*s,0.85f*s,LEAVES);
}

static void buildBush(float x, float z, float s)
{
    addEllipsoid(x,0.45f*s,z,0.72f*s,0.48f*s,0.55f*s,LEAVES);
    addEllipsoid(x+0.42f*s,0.42f*s,z+0.12f*s,0.48f*s,0.40f*s,0.46f*s,LEAVES2);
}

static void buildLamp(float x, float z)
{
    addCylinder(x,0.0f,z,0.09f,4.4f,METAL);
    addBox(x,4.45f,z,0.72f,0.16f,0.72f,METAL);
    addBox(x,4.32f,z,0.52f,0.10f,0.52f,LAMP_COLOR);
}

static void buildMailbox(float x, float z)
{
    addCylinder(x,0.0f,z,0.07f,1.05f,METAL);
    addBox(x,1.03f,z,0.78f,0.48f,0.48f,METAL);
    addBox(x,1.05f,z-0.25f,0.48f,0.18f,0.025f,BLACK);
    addBox(x,1.30f,z,0.08f,0.18f,0.08f,WOOD_LIGHT);
}

static void buildFence(float x, float z, float length)
{
    addBox(x,0.65f,z,length,0.12f,0.12f,WOOD);
    addBox(x,1.12f,z,length,0.12f,0.12f,WOOD);

    int count=(int)(length/1.35f);
    for (int i=0; i<=count; ++i)
    {
        float px=x-length*0.5f+i*1.35f;
        addBox(px,0.0f,z,0.14f,1.35f,0.14f,WOOD_LIGHT);
        addCone(px,1.32f,z,0.11f,0.16f,WOOD_LIGHT);
    }
}

// ============================================================
// HOUSE DETAILS
// Houses face the road. House centers are at x = +/- 17.5.
// ============================================================

static void buildWindowOnFront(float frontX, float y, float z, bool facesRight)
{
    float sign=facesRight ? 1.0f : -1.0f;

    // Window frame, glass, central mullion and sill.
    addBox(frontX,y,z,0.16f,1.45f,1.55f,TRIM);
    addBox(frontX+sign*0.095f,y,z,0.055f,1.20f,1.28f,GLASS);
    addBox(frontX+sign*0.13f,y,z,0.035f,1.18f,0.07f,TRIM);
    addBox(frontX+sign*0.13f,y,z,0.035f,0.07f,1.25f,TRIM);
    addBox(frontX+sign*0.14f,y-0.78f,z,0.25f,0.12f,1.75f,WHITE);
}

static void buildFrontDoor(float frontX, float z, bool facesRight, const float *doorColor)
{
    float sign=facesRight ? 1.0f : -1.0f;

    addBox(frontX,1.15f,z,0.20f,2.30f,1.35f,TRIM);
    addBox(frontX+sign*0.12f,1.15f,z,0.10f,2.05f,1.10f,doorColor);

    // Raised door panels.
    addBox(frontX+sign*0.18f,1.65f,z,0.035f,0.55f,0.72f,WOOD_LIGHT);
    addBox(frontX+sign*0.18f,0.85f,z,0.035f,0.55f,0.72f,WOOD_LIGHT);

    // Door handle.
    addCylinder(frontX+sign*0.23f,1.12f,z+0.38f,0.045f,0.10f,LAMP_COLOR);
}

static void buildGableRoof(
    float x, float ridgeY, float z,
    const float *roofColor)
{
    // The roof ridge runs along the length of the house.
    float a[3]={x-4.25f,ridgeY-1.8f,z-3.9f};
    float b[3]={x,ridgeY,z-3.9f};
    float c[3]={x,ridgeY,z+3.9f};
    float d[3]={x-4.25f,ridgeY-1.8f,z+3.9f};

    float e[3]={x,ridgeY,z-3.9f};
    float f[3]={x+4.25f,ridgeY-1.8f,z-3.9f};
    float g[3]={x+4.25f,ridgeY-1.8f,z+3.9f};
    float h[3]={x,ridgeY,z+3.9f};

    addQuad(a,b,c,d,roofColor);
    addQuad(e,f,g,h,roofColor);

    addBox(x,ridgeY-0.04f,z,0.20f,0.18f,8.1f,BLACK);
    addBox(x-4.25f,ridgeY-1.8f,z,0.18f,0.20f,8.1f,TRIM);
    addBox(x+4.25f,ridgeY-1.8f,z,0.18f,0.20f,8.1f,TRIM);
}

static void buildHouse(int number, float x, float z, bool leftSide)
{
    const float *wall=SIDING;
    const float *roof=ROOF;
    const float *door=WOOD;

    if (number==2)
    {
        wall=BRICK;
        roof=ROOF_BROWN;
        door=WOOD_LIGHT;
    }
    else if (number==3)
    {
        wall=BLUE_SIDING;
        roof=ROOF;
        door=BLACK;
    }
    else if (number==4)
    {
        wall=GREY_SIDING;
        roof=ROOF_BROWN;
        door=WOOD;
    }

    bool facesRight=leftSide;
    float frontX=leftSide ? x+3.95f : x-3.95f;
    float direction=leftSide ? 1.0f : -1.0f;

    // Main structure and concrete foundation.
    addBox(x,0.15f,z,7.8f,4.2f,6.8f,wall);
    addBox(x,0.0f,z,8.0f,0.32f,7.0f,CONCRETE);

    // Base trim and upper fascia.
    addBox(x,0.34f,z,7.95f,0.18f,6.95f,TRIM);
    addBox(x,4.05f,z,7.95f,0.18f,6.95f,TRIM);

    // Horizontal siding courses on houses 1, 3 and 4.
    if (number!=2)
    {
        for (int row=0; row<9; ++row)
        {
            float yy=0.65f+row*0.38f;
            addBox(x,yy,z,7.86f,0.035f,6.86f,WHITE);
        }
    }
    else
    {
        // Brick-like horizontal bands on house 2.
        for (int row=0; row<9; ++row)
        {
            float yy=0.58f+row*0.42f;
            addBox(frontX,yy,z,0.035f,0.035f,6.7f,BRICK_DARK);
        }
    }

    // Corner boards.
    addBox(x-3.78f,0.4f,z,0.16f,3.8f,0.16f,TRIM);
    addBox(x+3.78f,0.4f,z,0.16f,3.8f,0.16f,TRIM);
    addBox(x,0.4f,z-3.30f,7.6f,3.8f,0.14f,TRIM);
    addBox(x,0.4f,z+3.30f,7.6f,3.8f,0.14f,TRIM);

    // Street-facing door and two front windows.
    buildFrontDoor(frontX,z-0.05f,facesRight,door);
    buildWindowOnFront(frontX,z==0.0f?2.55f:2.55f,z-2.15f,facesRight);
    buildWindowOnFront(frontX,2.55f,z+2.15f,facesRight);

    // Side windows visible from the street at an angle.
    addBox(x,2.45f,z-3.48f,1.45f,1.35f,0.12f,TRIM);
    addBox(x,2.45f,z-3.40f,1.20f,1.10f,0.08f,GLASS);
    addBox(x,2.45f,z+3.48f,1.45f,1.35f,0.12f,TRIM);
    addBox(x,2.45f,z+3.40f,1.20f,1.10f,0.08f,GLASS);

    // Roof, gutters and downspouts.
    buildGableRoof(x,6.15f,z,roof);
    addBox(x,4.25f,z-3.48f,8.2f,0.12f,0.15f,METAL);
    addBox(x,4.25f,z+3.48f,8.2f,0.12f,0.15f,METAL);
    addBox(x+direction*3.65f,2.2f,z-3.30f,0.10f,3.8f,0.10f,METAL);

    // Porch roof and porch deck.
    addBox(frontX+direction*1.0f,2.95f,z,2.2f,0.18f,4.0f,roof);
    addBox(frontX+direction*1.0f,0.22f,z,2.5f,0.22f,4.2f,WOOD_LIGHT);
    addBox(frontX+direction*1.0f,0.40f,z,2.5f,0.12f,4.2f,TRIM);

    // Porch posts.
    for (int side=-1; side<=1; side+=2)
    {
        addBox(frontX+direction*1.95f,0.35f,z+side*1.75f,
               0.16f,2.65f,0.16f,TRIM);
    }

    // Steps from porch to the yard.
    addBox(frontX+direction*2.25f,0.10f,z,1.2f,0.18f,2.2f,CONCRETE2);

    // Small number marker beside the front door.
    addBox(frontX+direction*0.18f,2.35f,z+0.95f,
           0.08f,0.38f,0.32f,BLACK);
    addBox(frontX+direction*0.23f,2.35f,z+0.95f,
           0.035f,0.26f,0.22f,LAMP_COLOR);

    // House 3 has an upper dormer-like feature.
    if (number==3)
    {
        addBox(x,5.0f,z,2.2f,1.5f,2.0f,BLUE_SIDING);
        addBox(x+direction*1.12f,5.0f,z,0.12f,0.9f,1.1f,TRIM);
        addBox(x+direction*1.20f,5.0f,z,0.06f,0.7f,0.8f,GLASS);
        addBox(x,5.85f,z,2.8f,0.16f,2.5f,ROOF);
    }

    // House 4 gets a slightly weathered side extension.
    if (number==4)
    {
        addBox(x-direction*4.7f,0.12f,z+1.0f,1.8f,3.0f,3.4f,BRICK_DARK);
        addBox(x-direction*4.7f,3.15f,z+1.0f,2.0f,0.16f,3.6f,ROOF_BROWN);
    }
}

// ============================================================
// COMPLETE NEIGHBORHOOD
// ============================================================

static void buildNeighborhood()
{
    buildGround();
    buildRoad();

    // Four prototypes. Two on each side of the road.
    buildHouse(1,-17.5f,-13.0f,true);
    buildHouse(2, 17.5f,-13.0f,false);
    buildHouse(3,-17.5f, 13.0f,true);
    buildHouse(4, 17.5f, 13.0f,false);

    // Driveways leading toward each house.
    buildDriveway(-12.0f,-13.0f);
    buildDriveway( 12.0f,-13.0f);
    buildDriveway(-12.0f, 13.0f);
    buildDriveway( 12.0f, 13.0f);

    // Front garden bushes.
    buildBush(-12.5f,-16.0f,1.0f);
    buildBush(-12.5f,-10.0f,0.9f);
    buildBush( 12.5f,-16.0f,0.9f);
    buildBush( 12.5f,-10.0f,1.0f);
    buildBush(-12.5f, 10.0f,1.0f);
    buildBush(-12.5f, 16.0f,0.9f);
    buildBush( 12.5f, 10.0f,0.9f);
    buildBush( 12.5f, 16.0f,1.0f);

    // Short front-yard fences with gaps for driveways.
    buildFence(-17.5f,-18.0f,7.5f);
    buildFence( 17.5f,-18.0f,7.5f);
    buildFence(-17.5f, 18.0f,7.5f);
    buildFence( 17.5f, 18.0f,7.5f);

    // Street trees, spaced away from the temporary spawn.
    buildTree(-10.5f,-27.0f,1.0f);
    buildTree( 10.5f,-27.0f,1.0f);
    buildTree(-10.5f, 27.0f,1.0f);
    buildTree( 10.5f, 27.0f,1.0f);
    buildTree(-22.0f,  0.0f,1.1f);
    buildTree( 22.0f,  0.0f,1.1f);

    // Streetlights on the sidewalk edges.
    buildLamp(-8.7f,-24.0f);
    buildLamp( 8.7f,-24.0f);
    buildLamp(-8.7f, 24.0f);
    buildLamp( 8.7f, 24.0f);

    // Mailboxes at each property.
    buildMailbox(-11.5f,-17.0f);
    buildMailbox( 11.5f,-17.0f);
    buildMailbox(-11.5f, 17.0f);
    buildMailbox( 11.5f, 17.0f);

    // Small flower beds near the front gardens.
    for (int i=0; i<8; ++i)
    {
        float z=-15.0f+i*0.55f;
        addEllipsoid(-13.0f,0.18f,z,0.16f,0.13f,0.16f,FLOWER);
        addEllipsoid( 13.0f,0.18f,z,0.16f,0.13f,0.16f,FLOWER);
    }
    for (int i=0; i<8; ++i)
    {
        float z=11.0f+i*0.55f;
        addEllipsoid(-13.0f,0.18f,z,0.16f,0.13f,0.16f,FLOWER);
        addEllipsoid( 13.0f,0.18f,z,0.16f,0.13f,0.16f,FLOWER);
    }
}

// ============================================================
// COLLISION
// Prevent walking through the four house footprints.
// ============================================================

static bool insideHouse(float x, float z)
{
    const float centersX[4]={-17.5f,17.5f,-17.5f,17.5f};
    const float centersZ[4]={-13.0f,-13.0f,13.0f,13.0f};

    for (int i=0; i<4; ++i)
    {
        if (fabsf(x-centersX[i]) < 4.25f &&
            fabsf(z-centersZ[i]) < 3.75f)
            return true;
    }
    return false;
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

static void drawScene(void *user)
{
    SceneView *scene=(SceneView *)user;

    GX2SetDepthOnlyControl(TRUE,TRUE,GX2_COMPARE_FUNC_LESS);
    GX2SetCullOnlyControl(GX2_FRONT_FACE_CCW,FALSE,FALSE);

    CremaShaderBind(scene->shader);

    GX2SetVertexUniformBlock(
        scene->vertexBlock,sizeof(GlobalBlock),scene->uniforms);

    GX2SetPixelUniformBlock(
        scene->pixelBlock,sizeof(GlobalBlock),scene->uniforms);

    GX2RSetAttributeBuffer(
        scene->vertexBuffer,0,sizeof(Vertex),0);

    GX2RDrawIndexed(
        GX2_PRIMITIVE_MODE_TRIANGLES,
        scene->indexBuffer,
        GX2_INDEX_TYPE_U16,
        scene->indexCount,
        0,0,1);
}

// ============================================================
// GAMEPAD ORIENTATION AND CALIBRATION
// ============================================================

struct PadBasis
{
    Vec3 x;
    Vec3 y;
    Vec3 z;
};

static Vec3 baseX={1.0f,0.0f,0.0f};
static Vec3 baseY={0.0f,1.0f,0.0f};
static Vec3 baseZ={0.0f,0.0f,1.0f};

static Vec3 calibrationSumX={0.0f,0.0f,0.0f};
static Vec3 calibrationSumY={0.0f,0.0f,0.0f};
static Vec3 calibrationSumZ={0.0f,0.0f,0.0f};

static int calibrationSamples=0;
static const int CALIBRATION_SAMPLE_COUNT=90;

static bool calibrationActive=true;
static bool orientationCalibrated=false;

static float cameraYaw=0.0f;
static float cameraPitch=0.0f;

static bool readGamePad(VPADStatus *status)
{
    VPADReadError error;
    return VPADRead(VPAD_CHAN_0,status,1,&error)>0;
}

static float dot3(Vec3 a, Vec3 b)
{
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

static Vec3 normalize3(Vec3 v)
{
    float len=sqrtf(dot3(v,v));
    if (len<0.0001f)
        return {0.0f,0.0f,0.0f};

    return {v.x/len,v.y/len,v.z/len};
}

static Vec3 cross3(Vec3 a, Vec3 b)
{
    return {
        a.y*b.z-a.z*b.y,
        a.z*b.x-a.x*b.z,
        a.x*b.y-a.y*b.x
    };
}

static bool getPadBasis(const VPADStatus &pad, PadBasis *basis)
{
    basis->x=normalize3({pad.direction.x.x,pad.direction.x.y,pad.direction.x.z});
    basis->y=normalize3({pad.direction.y.x,pad.direction.y.y,pad.direction.y.z});
    basis->z=normalize3({pad.direction.z.x,pad.direction.z.y,pad.direction.z.z});

    if (dot3(basis->x,basis->x)<0.5f ||
        dot3(basis->y,basis->y)<0.5f ||
        dot3(basis->z,basis->z)<0.5f)
        return false;

    return true;
}

static void startCalibration()
{
    calibrationSumX={0.0f,0.0f,0.0f};
    calibrationSumY={0.0f,0.0f,0.0f};
    calibrationSumZ={0.0f,0.0f,0.0f};

    calibrationSamples=0;
    calibrationActive=true;
    orientationCalibrated=false;
    cameraYaw=0.0f;
    cameraPitch=0.0f;
}

static void addCalibrationSample(const PadBasis &sample)
{
    calibrationSumX.x+=sample.x.x;
    calibrationSumX.y+=sample.x.y;
    calibrationSumX.z+=sample.x.z;

    calibrationSumY.x+=sample.y.x;
    calibrationSumY.y+=sample.y.y;
    calibrationSumY.z+=sample.y.z;

    calibrationSumZ.x+=sample.z.x;
    calibrationSumZ.y+=sample.z.y;
    calibrationSumZ.z+=sample.z.z;

    calibrationSamples++;

    if (calibrationSamples<CALIBRATION_SAMPLE_COUNT)
        return;

    baseX=normalize3(calibrationSumX);
    Vec3 averageY=normalize3(calibrationSumY);
    Vec3 averageZ=normalize3(calibrationSumZ);

    float projection=dot3(baseX,averageY);

    baseY=normalize3({
        averageY.x-baseX.x*projection,
        averageY.y-baseX.y*projection,
        averageY.z-baseX.z*projection
    });

    baseZ=normalize3(cross3(baseX,baseY));

    if (dot3(baseZ,averageZ)<0.0f)
    {
        baseZ.x=-baseZ.x;
        baseZ.y=-baseZ.y;
        baseZ.z=-baseZ.z;
    }

    calibrationActive=false;
    orientationCalibrated=true;
    cameraYaw=0.0f;
    cameraPitch=0.0f;
}

static bool getPadAngles(
    const PadBasis &current,
    float *yaw,
    float *pitch)
{
    float fx=dot3(baseX,current.y);
    float fy=dot3(baseY,current.y);
    float fz=dot3(baseZ,current.y);

    float horizontal=sqrtf(fx*fx+fy*fy);
    if (horizontal<0.05f)
        return false;

    // Reverse horizontal body-turn tracking so turning your body left
    // makes the in-game view turn left. Analog-stick movement is unchanged.
    *yaw=-atan2f(fx,fy);
    *pitch=atan2f(fz,horizontal);
    return true;
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    if (!CremaAppInit("The Park"))
        return -1;

    if (!CremaShaderInitCompiler())
    {
        CremaAppShutdown();
        return -1;
    }

    buildNeighborhood();

    if (vertices.empty() || indices.empty())
    {
        CremaShaderShutdownCompiler();
        CremaAppShutdown();
        return -1;
    }

    const CremaAttrib attributes[] =
    {
        {0,0,offsetof(Vertex,position),GX2_ATTRIB_FORMAT_FLOAT_32_32_32},
        {1,0,offsetof(Vertex,normal),GX2_ATTRIB_FORMAT_FLOAT_32_32_32},
        {2,0,offsetof(Vertex,color),GX2_ATTRIB_FORMAT_FLOAT_32_32_32}
    };

    CremaShader *shader=CremaShaderCompile(
        VERTEX_SHADER,PIXEL_SHADER,attributes,3);

    if (!shader)
    {
        CremaShaderShutdownCompiler();
        CremaAppShutdown();
        return -1;
    }

    int32_t vertexBlock=CremaShaderVSBlockLocation(shader,"Global");
    int32_t pixelBlock=CremaShaderPSBlockLocation(shader,"Global");

    if (vertexBlock<0) vertexBlock=0;
    if (pixelBlock<0) pixelBlock=0;

    GX2RBuffer vertexBuffer{};
    GX2RBuffer indexBuffer{};

    bool buffersReady=
        CremaBufferCreateVertex(
            &vertexBuffer,sizeof(Vertex),
            (uint32_t)vertices.size(),vertices.data())
        &&
        CremaBufferCreateIndexU16(
            &indexBuffer,(uint32_t)indices.size(),indices.data());

    if (!buffersReady)
    {
        CremaBufferDestroy(&vertexBuffer);
        CremaBufferDestroy(&indexBuffer);
        CremaShaderFree(shader);
        CremaShaderShutdownCompiler();
        CremaAppShutdown();
        return -1;
    }

    CremaUniformRing globals{};

    if (!CremaUniformRingCreate(
        &globals,sizeof(GlobalBlock),CREMA_FRAMES_IN_FLIGHT))
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

    // Temporary test spawn: exact center of the street.
    Vec3 camPos={0.0f,1.7f,0.0f};

    Mat4 projection=mat4_perspective(
        60.0f*PI/180.0f,
        16.0f/9.0f,
        0.1f,
        150.0f);

    Vec3 lightRaw={-0.45f,-0.80f,-0.35f};
    Vec3 light=vec3_normalize(lightRaw);

    SceneView scene{};
    scene.shader=shader;
    scene.vertexBuffer=&vertexBuffer;
    scene.indexBuffer=&indexBuffer;
    scene.vertexBlock=vertexBlock;
    scene.pixelBlock=pixelBlock;
    scene.indexCount=(uint32_t)indices.size();

    startCalibration();
    bool plusWasPressed=false;

    while (CremaAppRunning())
    {
        CremaClockTick(&clock);
        float dt=clampf(clock.dt,0.0f,0.05f);

        VPADStatus pad{};

        if (readGamePad(&pad))
        {
            PadBasis current{};
            bool validOrientation=getPadBasis(pad,&current);

            bool plusPressed=(pad.trigger&VPAD_BUTTON_PLUS)!=0;
            if (plusPressed && !plusWasPressed)
                startCalibration();

            plusWasPressed=plusPressed;

            if (validOrientation)
            {
                if (calibrationActive)
                {
                    addCalibrationSample(current);
                }
                else if (orientationCalibrated)
                {
                    float measuredYaw=0.0f;
                    float measuredPitch=0.0f;

                    if (getPadAngles(current,&measuredYaw,&measuredPitch))
                    {
                        measuredPitch=-measuredPitch;
                        measuredPitch=clampf(measuredPitch,-1.2f,1.2f);

                        float smoothing=clampf(dt*8.0f,0.0f,1.0f);

                        cameraYaw+=angleDifference(measuredYaw,cameraYaw)*smoothing;
                        cameraPitch+=(measuredPitch-cameraPitch)*smoothing;
                    }
                }
            }

            float lx=pad.leftStick.x;
            float ly=pad.leftStick.y;

            if (fabsf(lx)<0.12f) lx=0.0f;
            if (fabsf(ly)<0.12f) ly=0.0f;

            float forwardX=-sinf(cameraYaw);
            float forwardZ=-cosf(cameraYaw);
            float rightX=cosf(cameraYaw);
            float rightZ=-sinf(cameraYaw);

            const float walkingSpeed=4.5f;

            if (!calibrationActive)
            {
                float nextX=camPos.x+
                    (forwardX*ly+rightX*lx)*walkingSpeed*dt;

                float nextZ=camPos.z+
                    (forwardZ*ly+rightZ*lx)*walkingSpeed*dt;

                // Test each axis separately so the player can slide along walls.
                if (!insideHouse(nextX,camPos.z))
                    camPos.x=nextX;

                if (!insideHouse(camPos.x,nextZ))
                    camPos.z=nextZ;
            }
        }

        // Keep the player inside the test neighborhood.
        camPos.x=clampf(camPos.x,-38.0f,38.0f);
        camPos.z=clampf(camPos.z,-38.0f,38.0f);

        uint32_t slot=CremaFrameBegin(&frame);

        Mat4 view=mat4_mul(
            mat4_rotate_x(-cameraPitch),
            mat4_mul(
                mat4_rotate_y(-cameraYaw),
                mat4_translate(-camPos.x,-camPos.y,-camPos.z)));

        GlobalBlock block{};
        block.viewProj=mat4_mul(projection,view);

        block.lightDir[0]=light.x;
        block.lightDir[1]=light.y;
        block.lightDir[2]=light.z;
        block.lightDir[3]=0.0f;

        block.camPos[0]=camPos.x;
        block.camPos[1]=camPos.y;
        block.camPos[2]=camPos.z;
        block.camPos[3]=0.0f;

        block.fogParam[0]=58.0f;
        block.fogParam[1]=105.0f;

        block.fogColor[0]=0.48f;
        block.fogColor[1]=0.68f;
        block.fogColor[2]=0.82f;
        block.fogColor[3]=1.0f;

        scene.uniforms=CremaUniformRingStore(
            &globals,slot,&block,sizeof(block));

        const float sky[4]={0.48f,0.68f,0.82f,1.0f};

        CremaFrameDrawBoth(
            sky,drawScene,&scene);

        CremaFrameEnd(&frame,&stats);
    }

    // Wait for submitted GPU work before freeing graphics resources.
    CremaFrameSettle(&frame);

    CremaUniformRingDestroy(&globals);
    CremaBufferDestroy(&indexBuffer);
    CremaBufferDestroy(&vertexBuffer);

    CremaShaderFree(shader);
    CremaShaderShutdownCompiler();
    CremaAppShutdown();

    return 0;
}
