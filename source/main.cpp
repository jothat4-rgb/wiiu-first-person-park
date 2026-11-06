#include <gx2/draw.h>
#include <gx2/state.h>
#include <gx2/registers.h>
#include <gx2/sampler.h>
#include <gx2/texture.h>
#include <vpad/input.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include "crema_app.h"
#include "crema_frame.h"
#include "crema_matrix.h"
#include "crema_mesh.h"
#include "crema_shader.h"
#include "crema_texture.h"

// ============================================================
// THE PARK - KENNEY ASSET RENDERER
//
// Expected content files:
// roads.cmesh / roads.ctex
// suburban.cmesh / suburban.ctex
// nature.cmesh / nature.ctex
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

#define GLOBAL_BLOCK \
    "layout(binding=0) uniform Global {\n" \
    " mat4 uViewProj;\n" \
    " vec4 uLightDir;\n" \
    " vec4 uCamPos;\n" \
    " vec4 uFogParam;\n" \
    " vec4 uFogColor;\n" \
    "};\n"

static const char *VERTEX_SHADER =
    "#version 450\n"
    "layout(location=0) in vec3 aPosition;\n"
    "layout(location=1) in vec3 aNormal;\n"
    "layout(location=2) in vec2 aUV;\n"
    GLOBAL_BLOCK
    "layout(location=0) out vec2 vUV;\n"
    "layout(location=1) out vec3 vNormal;\n"
    "layout(location=2) out vec3 vWorld;\n"
    "void main(){\n"
    " gl_Position=uViewProj*vec4(aPosition,1.0);\n"
    " vUV=aUV;\n"
    " vNormal=aNormal;\n"
    " vWorld=aPosition;\n"
    "}\n";

static const char *PIXEL_SHADER =
    "#version 450\n"
    "layout(location=0) in vec2 vUV;\n"
    "layout(location=1) in vec3 vNormal;\n"
    "layout(location=2) in vec3 vWorld;\n"
    GLOBAL_BLOCK
    "layout(binding=0) uniform sampler2D uTexture;\n"
    "layout(location=0) out vec4 oColor;\n"
    "void main(){\n"
    " vec4 tex=texture(uTexture,vUV);\n"
    " vec3 n=normalize(vNormal);\n"
    " float diffuse=max(dot(n,-uLightDir.xyz),0.0);\n"
    " float light=0.30+diffuse*0.70;\n"
    " vec3 color=tex.rgb*light;\n"
    " float distanceToCamera=length(vWorld-uCamPos.xyz);\n"
    " float fog=smoothstep(uFogParam.x,uFogParam.y,distanceToCamera);\n"
    " color=mix(color,uFogColor.rgb,fog);\n"
    " oColor=vec4(color,tex.a);\n"
    "}\n";

// ============================================================
// GLOBAL UNIFORM DATA
// ============================================================

struct GlobalBlock
{
    Mat4 viewProj;
    float lightDir[4];
    float camPos[4];
    float fogParam[4];
    float fogColor[4];
};

struct WorldAsset
{
    CremaMesh mesh;
    GX2Texture texture;
    bool loaded;
};

struct DrawContext
{
    CremaShader *shader;
    WorldAsset *assets;
    GX2Sampler *sampler;
    const void *uniforms;
    int32_t vertexBlock;
    int32_t pixelBlock;
    uint32_t textureUnit;
};

// ============================================================
// DRAW THE THREE ASSET GROUPS
// ============================================================

static void drawWorld(void *user)
{
    DrawContext *ctx=(DrawContext *)user;

    GX2SetDepthOnlyControl(
        TRUE,TRUE,GX2_COMPARE_FUNC_LESS);

    GX2SetCullOnlyControl(
        GX2_FRONT_FACE_CCW,FALSE,FALSE);

    CremaShaderBind(ctx->shader);

    GX2SetVertexUniformBlock(
        ctx->vertexBlock,
        sizeof(GlobalBlock),
        ctx->uniforms);

    GX2SetPixelUniformBlock(
        ctx->pixelBlock,
        sizeof(GlobalBlock),
        ctx->uniforms);

    for (int i=0;i<3;i++)
    {
        if (!ctx->assets[i].loaded)
            continue;

        GX2SetPixelTexture(
            &ctx->assets[i].texture,
            ctx->textureUnit);

        GX2SetPixelSampler(
            ctx->sampler,
            ctx->textureUnit);

        CremaMeshDraw(
            &ctx->assets[i].mesh,1);
    }
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

static float dot3(Vec3 a,Vec3 b)
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

static Vec3 cross3(Vec3 a,Vec3 b)
{
    return {
        a.y*b.z-a.z*b.y,
        a.z*b.x-a.x*b.z,
        a.x*b.y-a.y*b.x
    };
}

static bool getPadBasis(
    const VPADStatus &pad,
    PadBasis *basis)
{
    basis->x=normalize3({
        pad.direction.x.x,
        pad.direction.x.y,
        pad.direction.x.z
    });

    basis->y=normalize3({
        pad.direction.y.x,
        pad.direction.y.y,
        pad.direction.y.z
    });

    basis->z=normalize3({
        pad.direction.z.x,
        pad.direction.z.y,
        pad.direction.z.z
    });

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

    // Left/right direction is intentionally reversed here
    // so physical left turns the in-game view left.
    *yaw=-atan2f(fx,fy);
    *pitch=atan2f(fz,horizontal);

    return true;
}

// ============================================================
// LOAD AND RELEASE ASSETS
// ============================================================

static bool loadWorldAssets(WorldAsset *assets)
{
    const char *meshPaths[3]=
    {
        "/vol/content/roads.cmesh",
        "/vol/content/suburban.cmesh",
        "/vol/content/nature.cmesh"
    };

    const char *texturePaths[3]=
    {
        "/vol/content/roads.ctex",
        "/vol/content/suburban.ctex",
        "/vol/content/nature.ctex"
    };

    for (int i=0;i<3;i++)
    {
        assets[i].loaded=false;

        if (!CremaMeshLoad(
            &assets[i].mesh,meshPaths[i]))
            return false;

        if (!CremaTextureLoad(
            &assets[i].texture,texturePaths[i]))
        {
            CremaMeshDestroy(&assets[i].mesh);
            return false;
        }

        assets[i].loaded=true;
    }

    return true;
}

static void releaseWorldAssets(WorldAsset *assets)
{
    for (int i=0;i<3;i++)
    {
        if (!assets[i].loaded)
            continue;

        CremaTextureDestroy(&assets[i].texture);
        CremaMeshDestroy(&assets[i].mesh);

        assets[i].loaded=false;
    }
}

// ============================================================
// MAIN
// ============================================================

int main(int argc,char **argv)
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

    WorldAsset assets[3]{};

    if (!loadWorldAssets(assets))
    {
        releaseWorldAssets(assets);
        CremaShaderShutdownCompiler();
        CremaAppShutdown();
        return -1;
    }

    // All three baked meshes use the same position/normal/UV layout.
    CremaShader *shader=CremaShaderCompile(
        VERTEX_SHADER,
        PIXEL_SHADER,
        assets[0].mesh.attribs,
        assets[0].mesh.attribCount);

    if (!shader)
    {
        releaseWorldAssets(assets);
        CremaShaderShutdownCompiler();
        CremaAppShutdown();
        return -1;
    }

    int32_t vertexBlock=
        CremaShaderVSBlockLocation(shader,"Global");

    int32_t pixelBlock=
        CremaShaderPSBlockLocation(shader,"Global");

    if (vertexBlock<0) vertexBlock=0;
    if (pixelBlock<0) pixelBlock=0;

    uint32_t textureUnit=0;

    if (shader->ps->samplerVarCount>0)
        textureUnit=shader->ps->samplerVars[0].location;

    GX2Sampler sampler{};
    CremaSamplerInitTrilinear(
        &sampler,GX2_TEX_CLAMP_MODE_WRAP);

    CremaUniformRing globals{};

    if (!CremaUniformRingCreate(
        &globals,sizeof(GlobalBlock),CREMA_FRAMES_IN_FLIGHT))
    {
        CremaShaderFree(shader);
        releaseWorldAssets(assets);
        CremaShaderShutdownCompiler();
        CremaAppShutdown();
        return -1;
    }

    CremaFrame frame;
    CremaFrameInit(
        &frame,CREMA_PACING_FENCED,1);

    CremaClock clock;
    CremaClockInit(&clock);

    CremaFrameStats stats{};

    // Spawn in the exact middle of the street.
    Vec3 camPos={0.0f,1.7f,0.0f};

    Mat4 projection=mat4_perspective(
        60.0f*PI/180.0f,
        16.0f/9.0f,
        0.1f,
        180.0f);

    Vec3 lightRaw={-0.45f,-0.80f,-0.35f};
    Vec3 light=vec3_normalize(lightRaw);

    DrawContext scene{};
    scene.shader=shader;
    scene.assets=assets;
    scene.sampler=&sampler;
    scene.vertexBlock=vertexBlock;
    scene.pixelBlock=pixelBlock;
    scene.textureUnit=textureUnit;

    startCalibration();

    bool plusWasPressed=false;

    while (CremaAppRunning())
    {
        CremaClockTick(&clock);

        float dt=clampf(clock.dt,0.0f,0.05f);

        VPADStatus pad{};

        if (readGamePad(&pad))
        {
            bool leftHeld=
                (pad.hold&VPAD_BUTTON_L)!=0;

            bool rightHeld=
                (pad.hold&VPAD_BUTTON_R)!=0;

            bool plusHeld=
                (pad.hold&VPAD_BUTTON_PLUS)!=0;

            // Emergency exit: hold L + R + Plus.
            if (leftHeld && rightHeld && plusHeld)
                break;

            bool plusPressed=
                (pad.trigger&VPAD_BUTTON_PLUS)!=0;

            if (plusPressed && !plusWasPressed)
                startCalibration();

            plusWasPressed=plusPressed;

            PadBasis current{};

            if (getPadBasis(pad,&current))
            {
                if (calibrationActive)
                {
                    addCalibrationSample(current);
                }
                else if (orientationCalibrated)
                {
                    float measuredYaw=0.0f;
                    float measuredPitch=0.0f;

                    if (getPadAngles(
                        current,&measuredYaw,&measuredPitch))
                    {
                        measuredPitch=-measuredPitch;
                        measuredPitch=clampf(
                            measuredPitch,-1.2f,1.2f);

                        float smoothing=
                            clampf(dt*8.0f,0.0f,1.0f);

                        cameraYaw+=
                            angleDifference(
                                measuredYaw,cameraYaw)*smoothing;

                        cameraPitch+=
                            (measuredPitch-cameraPitch)*smoothing;
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
                camPos.x+=
                    (forwardX*ly+rightX*lx)*
                    walkingSpeed*dt;

                camPos.z+=
                    (forwardZ*ly+rightZ*lx)*
                    walkingSpeed*dt;
            }
        }

        camPos.x=clampf(camPos.x,-38.0f,38.0f);
        camPos.z=clampf(camPos.z,-38.0f,38.0f);

        uint32_t slot=CremaFrameBegin(&frame);

        Mat4 view=mat4_mul(
            mat4_rotate_x(-cameraPitch),
            mat4_mul(
                mat4_rotate_y(-cameraYaw),
                mat4_translate(
                    -camPos.x,-camPos.y,-camPos.z)));

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

        block.fogParam[0]=75.0f;
        block.fogParam[1]=135.0f;

        block.fogColor[0]=0.48f;
        block.fogColor[1]=0.68f;
        block.fogColor[2]=0.82f;
        block.fogColor[3]=1.0f;

        scene.uniforms=CremaUniformRingStore(
            &globals,slot,&block,sizeof(block));

        const float sky[4]=
        {
            0.48f,0.68f,0.82f,1.0f
        };

        CremaFrameDrawBoth(
            sky,drawWorld,&scene);

        CremaFrameEnd(&frame,&stats);
    }

    // Finish GPU work before releasing loaded resources.
    CremaFrameSettle(&frame);

    CremaUniformRingDestroy(&globals);

    CremaShaderFree(shader);

    releaseWorldAssets(assets);

    CremaShaderShutdownCompiler();
    CremaAppShutdown();

    return 0;
}
