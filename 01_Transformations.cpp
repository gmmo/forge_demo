/*
 * Copyright (c) 2017-2025 The Forge Interactive Inc.
 *
 * This file is part of The-Forge
 * (see https://github.com/ConfettiFX/The-Forge).
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// Cloth Physics Demo - Gustavo Oliveira
// Demonstrates cloth simulation with collision detection

// Interfaces
#include "../../../../Common_3/Application/Interfaces/IApp.h"
#include "../../../../Common_3/Application/Interfaces/ICamera.h"
#include "../../../../Common_3/Application/Interfaces/IFont.h"
#include "../../../../Common_3/Application/Interfaces/IProfiler.h"
#include "../../../../Common_3/Application/Interfaces/IScreenshot.h"
#include "../../../../Common_3/Application/Interfaces/IUI.h"
#include "../../../../Common_3/Game/Interfaces/IScripting.h"
#include "../../../../Common_3/Utilities/Interfaces/IFileSystem.h"
#include "../../../../Common_3/Utilities/Interfaces/ILog.h"
#include "../../../../Common_3/Utilities/Interfaces/ITime.h"

#include "../../../../Common_3/Utilities/RingBuffer.h"

// Renderer
#include "../../../../Common_3/Graphics/Interfaces/IGraphics.h"
#include "../../../../Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h"

// Math
#include "../../../../Common_3/Utilities/Interfaces/IMath.h"

#include "../../../../Common_3/Utilities/Interfaces/IMemory.h"

// fsl
#include "../../../../Common_3/Graphics/FSL/defaults.h"
#include "./Shaders/FSL/Global.srt.h"

#include "Common.h"
#include "Physics.h"
#include "Meshes.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//                                      Structs
//////////////////////////////////////////////////////////////////////////////////////////////////

struct UniformBlock
{
    CameraMatrix mProjectView;
    CameraMatrix mSkyProjectView;

     mat4         mToWorldMat;
     vec4         mColor;
     float        mGeometryWeight[4];

    // Point Light Information
    vec4 mLightPosition;
    vec4 mLightColor;
    vec4 mCameraPosition;
    vec4 mLightDirection;
};

// Single point light source
struct LightParams
{
    vec3  position = { -50.0f, 80.0f, 40.0f }; // high up, between camera and scene
    vec3  direction = { 0.0f, 0.90f, 0.50f };  // pointing down and toward cloth
    vec3  color = { 1.0f, 1.0f, 1.0f };
    float intensity = 1.0f;
};

//////////////////////////////////////////////////////////////////////////////////////////////////
//                                      Globals
//////////////////////////////////////////////////////////////////////////////////////////////////

// But we only need Two sets of resources (one in flight and one being used on CPU)
const uint32_t gDataBufferCount = 2;

static LightParams gLight;

Renderer*  pRenderer = NULL;
Queue*     pGraphicsQueue = NULL;
GpuCmdRing gGraphicsCmdRing = {};

SwapChain*    pSwapChain = NULL;
RenderTarget* pDepthBuffer = NULL;
Semaphore*    pImageAcquiredSemaphore[gDataBufferCount] = { NULL };

Shader*      pSphereShader = NULL;

Shader*        pSkyBoxDrawShader = NULL;
Buffer*        pSkyBoxVertexBuffer = NULL;
Pipeline*      pSkyBoxDrawPipeline = NULL;
Texture*       pSkyBoxTextures[6];
Sampler*       pSkyBoxSampler = {};
DescriptorSet* pDescriptorSetPersistent = { NULL };
DescriptorSet* pDescriptorSetPerFrame = { NULL };

Buffer* pUniformBuffer[gDataBufferCount][gNumObjects] = { NULL };

uint32_t     gFrameIndex = 0;
ProfileToken gGpuProfileToken = PROFILE_INVALID_TOKEN;

int              gNumberOfSpherePoints;
UniformBlock     gUniformData;


// VR 2D layer transform (positioned at -1 along the Z axis, default rotation, default scale)
VR2DLayerDesc    gVR2DLayer{ { 0.0f, 0.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 1.0f }, 1.0f };

ICamera* pCamera = NULL;

UIWindowDesc gGuiWindowDesc;
bool         gGuiActive = false;

uint32_t gFontID = 0;

QueryPool* pPipelineStatsQueryPool[gDataBufferCount] = {};

const char* pSkyBoxImageFileNames[] = {
    "pos_x.tex",         // Right (+X)
    "neg_x.tex",         // Left (-X)
    "pos_y.tex",         // Top (+Y) - Sky
    "texture_atlas.tex", // Bottom (-Y) - atlas has the texture for the scene
    "pos_z.tex",         // Front (+Z)
    "neg_z.tex"          // Back (-Z)
};

FontDrawDesc gFrameTimeDraw;

// Generate sky box vertex buffer
const float gSkyBoxPoints[] = {
    10.0f,  -10.0f, -10.0f, 6.0f, // -z
    -10.0f, -10.0f, -10.0f, 6.0f,   -10.0f, 10.0f,  -10.0f, 6.0f,   -10.0f, 10.0f,
    -10.0f, 6.0f,   10.0f,  10.0f,  -10.0f, 6.0f,   10.0f,  -10.0f, -10.0f, 6.0f,

    -10.0f, -10.0f, 10.0f,  2.0f, //-x
    -10.0f, -10.0f, -10.0f, 2.0f,   -10.0f, 10.0f,  -10.0f, 2.0f,   -10.0f, 10.0f,
    -10.0f, 2.0f,   -10.0f, 10.0f,  10.0f,  2.0f,   -10.0f, -10.0f, 10.0f,  2.0f,

    10.0f,  -10.0f, -10.0f, 1.0f, //+x
    10.0f,  -10.0f, 10.0f,  1.0f,   10.0f,  10.0f,  10.0f,  1.0f,   10.0f,  10.0f,
    10.0f,  1.0f,   10.0f,  10.0f,  -10.0f, 1.0f,   10.0f,  -10.0f, -10.0f, 1.0f,

    -10.0f, -10.0f, 10.0f,  5.0f, // +z
    -10.0f, 10.0f,  10.0f,  5.0f,   10.0f,  10.0f,  10.0f,  5.0f,   10.0f,  10.0f,
    10.0f,  5.0f,   10.0f,  -10.0f, 10.0f,  5.0f,   -10.0f, -10.0f, 10.0f,  5.0f,

    -10.0f, 10.0f,  -10.0f, 3.0f, //+y
    10.0f,  10.0f,  -10.0f, 3.0f,   10.0f,  10.0f,  10.0f,  3.0f,   10.0f,  10.0f,
    10.0f,  3.0f,   -10.0f, 10.0f,  10.0f,  3.0f,   -10.0f, 10.0f,  -10.0f, 3.0f,

    10.0f,  -10.0f, 10.0f,  4.0f, //-y
    10.0f,  -10.0f, -10.0f, 4.0f,   -10.0f, -10.0f, -10.0f, 4.0f,   -10.0f, -10.0f,
    -10.0f, 4.0f,   -10.0f, -10.0f, 10.0f,  4.0f,   10.0f,  -10.0f, 10.0f,  4.0f,
};

static unsigned char gPipelineStatsCharArray[2048] = {};
static bstring       gPipelineStats = bfromarr(gPipelineStatsCharArray);

const char* gWindowTestScripts[] = { "TestFullScreen.lua", "TestCenteredWindow.lua", "TestNonCenteredWindow.lua", "TestBorderless.lua" };

const char* gReloadServerTestScripts[] = { "TestReloadShader.lua", "TestReloadShaderCapture.lua" };

static void add_attribute(VertexLayout* layout, ShaderSemantic semantic, TinyImageFormat format, uint32_t offset)
{
    uint32_t n_attr = layout->mAttribCount++;

    VertexAttrib* attr = layout->mAttribs + n_attr;

    attr->mSemantic = semantic;
    attr->mFormat = format;
    attr->mBinding = 0;
    attr->mLocation = n_attr;
    attr->mOffset = offset;
}

static void copy_attribute(VertexLayout* layout, void* buffer_data, uint32_t offset, uint32_t size, uint32_t vcount, void* data)
{
    uint8_t* dst_data = static_cast<uint8_t*>(buffer_data);
    uint8_t* src_data = static_cast<uint8_t*>(data);
    for (uint32_t i = 0; i < vcount; ++i)
    {
        memcpy(dst_data + offset, src_data, size);

        dst_data += layout->mBindings[0].mStride;
        src_data += size;
    }
}

// Builds one GPU VB + IB from a given SimpleMesh into the given output pointers.
// Warning, the layout has to match the vertex morphing else the animation won't work
static void upload_mesh_to_gpu(
    const SimpleMesh& mesh,
    VertexLayout& layout,
    Buffer** ppVertexBuffer,
    Buffer** ppIndexBuffer,
    uint32_t& outIndexCount,
    void** ppCpuVertexData,
    size_t& outCpuVertexSize)
{
    layout = {};
    layout.mBindingCount = 1;
    layout.mBindings[0].mStride = 36;

    // Add attributes in memory order
    add_attribute(&layout, SEMANTIC_POSITION, TinyImageFormat_R32G32B32_SFLOAT, 0);
    add_attribute(&layout, SEMANTIC_TEXCOORD0, TinyImageFormat_R8G8B8A8_UNORM, 12);
    add_attribute(&layout, SEMANTIC_NORMAL, TinyImageFormat_R32G32B32_SFLOAT, 16);
    add_attribute(&layout, SEMANTIC_TEXCOORD4, TinyImageFormat_R32G32_SFLOAT, 28);

    size_t bufferSize = (size_t)mesh.vertexCount * layout.mBindings[0].mStride;
    void*  bufferData = tf_calloc(1, bufferSize);

    // Copy attributes with offsets and sizes
    copy_attribute(&layout, bufferData, 0, 12, mesh.vertexCount, mesh.verts);      // Position: 12 bytes
    copy_attribute(&layout, bufferData, 12, 4, mesh.vertexCount, mesh.colors);   // Color: 4 bytes
    copy_attribute(&layout, bufferData, 16, 12, mesh.vertexCount, mesh.normals); // Normal: 12 bytes
    copy_attribute(&layout, bufferData, 28, 8, mesh.vertexCount, mesh.UVs);     // UV: 8 bytes

    // Keep a CPU copy for the vertex-morph path
    outCpuVertexSize = bufferSize;
    *ppCpuVertexData = tf_malloc(bufferSize);
    memcpy(*ppCpuVertexData, bufferData, bufferSize);

    // Upload vertex buffer
    BufferLoadDesc vbDesc = {};
    vbDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
    vbDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
    vbDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
    vbDesc.mDesc.mSize = bufferSize;
    vbDesc.pData = bufferData;
    vbDesc.ppBuffer = ppVertexBuffer;
    addResource(&vbDesc, nullptr);

    // Upload index buffer
    size_t indexSize = sizeof(uint16_t) * mesh.indexCount;
    outIndexCount = mesh.indexCount;

    BufferLoadDesc ibDesc = {};
    ibDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_INDEX_BUFFER;
    ibDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
    ibDesc.mDesc.mSize = indexSize;
    ibDesc.pData = mesh.indices;
    ibDesc.ppBuffer = ppIndexBuffer;
    addResource(&ibDesc, nullptr);

    waitForAllResourceLoads();

    tf_free(bufferData);
}

// Create a rotation matrix that orients +Z axis toward 'direction'
static mat4 create_look_direction_matrix(vec3 position, vec3 direction)
{
    // Normalize the direction
    vec3 forward = normalize(-direction);

    vec3 worldUp = vec3(0.0f, 1.0f, 0.0f);
    if (fabsf(dot(forward, worldUp)) > 0.99f)
    {
        worldUp = vec3(1.0f, 0.0f, 0.0f);
    }

    vec3 right = normalize(cross(worldUp, forward));
    vec3 up = cross(forward, right);

    mat4 result;

    // Column 0 (right vector)
    result[0][0] = right.x;
    result[0][1] = right.y;
    result[0][2] = right.z;
    result[0][3] = 0.0f;

    // Column 1 (up vector)
    result[1][0] = up.x;
    result[1][1] = up.y;
    result[1][2] = up.z;
    result[1][3] = 0.0f;

    // Column 2 (forward vector - this is the direction the arrow points)
    result[2][0] = forward.x;
    result[2][1] = forward.y;
    result[2][2] = forward.z;
    result[2][3] = 0.0f;

    // Column 3 (translation/position)
    result[3][0] = position.x;
    result[3][1] = position.y;
    result[3][2] = position.z;
    result[3][3] = 1.0f;

    return result;
}
// Generates all meshes for the scene besided the skybox
// Cloth, light marker, and floor
static void generate_complex_mesh()
{
    // Hanging cloth with physics simulation
    const uint32_t CLOTH_X_SEGMENTS = 64;
    const uint32_t CLOTH_Z_SEGMENTS = 64;

    LOGF(eINFO, "Generating Object %u: Cloth", OBJ_CLOTH);
    generate_plane_mesh(
        gSimpleMesh[OBJ_CLOTH], 60.0f, 60.0f, CLOTH_X_SEGMENTS, CLOTH_Z_SEGMENTS,
        1,                                 // atlasSlot (cloth texture)
        PLANE_XZ, vec3(0.0f, 61.0f, 0.0f), // Position high up
        1.0f);                             // UV tiling

    upload_mesh_to_gpu(
        gSimpleMesh[OBJ_CLOTH], gMeshVertexLayout, &pMeshVertexBuffer[OBJ_CLOTH],
        &pMeshIndexBuffer[OBJ_CLOTH],
        gMeshIndexCount[OBJ_CLOTH],
        &gMeshCpuMappedVertexData[OBJ_CLOTH],
        gMeshCpuMappedVertexSize[OBJ_CLOTH]);
    copy_animation_data(OBJ_CLOTH, gSimpleMesh[OBJ_CLOTH]);

    // Transform and color
    gObjectWorld[OBJ_CLOTH] = mat4::identity();
    gObjectColor[OBJ_CLOTH] = vec4(1.0f, 1.0f, 1.0f, 1.0f);

    // Create cloth physics constraints
    gCloth.CreateFromPlaneMesh(gSimpleMesh[OBJ_CLOTH], CLOTH_X_SEGMENTS, CLOTH_Z_SEGMENTS);
    gCloth.SetSphereCollider(gSphereParams.position, gSphereParams.radius);

    LOGF(eINFO,
        "Cloth: %u particles, %u constraints", gCloth.GetParticleCount(), gCloth.GetConstraintCount());

    // Collision Sphere
    LOGF(eINFO,
        "Generating Object %u: Sphere", OBJ_SPHERE);
    generate_sphere_mesh(
        gSimpleMesh[OBJ_SPHERE],
        gSphereParams.radius,
        gSphereParams.segments,
        gSphereParams.rings, 0);

    upload_mesh_to_gpu(
        gSimpleMesh[OBJ_SPHERE],
        gMeshVertexLayout, &pMeshVertexBuffer[OBJ_SPHERE], &pMeshIndexBuffer[OBJ_SPHERE],
        gMeshIndexCount[OBJ_SPHERE], &gMeshCpuMappedVertexData[OBJ_SPHERE],
        gMeshCpuMappedVertexSize[OBJ_SPHERE]);
    copy_animation_data(OBJ_SPHERE, gSimpleMesh[OBJ_SPHERE]);

    // Transform and color
    gObjectWorld[OBJ_SPHERE] = mat4::translation(gSphereParams.position);
    gObjectColor[OBJ_SPHERE] = vec4(1.0f, 1.0f, 1.0f, 1.0f);

    LOGF(eINFO, "  Sphere: verts=%u indices=%u",
        gSimpleMesh[OBJ_SPHERE].vertexCount, gSimpleMesh[OBJ_SPHERE].indexCount);

    // Ligh marker pointing in light direction
    LOGF(eINFO, "Generating Object %u: Light Marker (Arrow)", OBJ_LIGHT_MARKER);
    generate_arrow_mesh(gSimpleMesh[OBJ_LIGHT_MARKER],
                        20.0f, // total length (was 8.0f) - 2.5x bigger
                        1.2f,  // shaft radius (was 0.5f) - 2.4x thicker
                        7.0f,  // tip length (was 3.0f) - 2.3x longer
                        3.5f,  // tip radius (was 1.5f) - 2.3x wider
                        3);    // checkerboard texture

    upload_mesh_to_gpu(
        gSimpleMesh[OBJ_LIGHT_MARKER], gMeshVertexLayout, &pMeshVertexBuffer[OBJ_LIGHT_MARKER],
        &pMeshIndexBuffer[OBJ_LIGHT_MARKER], gMeshIndexCount[OBJ_LIGHT_MARKER],
        &gMeshCpuMappedVertexData[OBJ_LIGHT_MARKER],
        gMeshCpuMappedVertexSize[OBJ_LIGHT_MARKER]);

    copy_animation_data(OBJ_LIGHT_MARKER, gSimpleMesh[OBJ_LIGHT_MARKER]);

    // Transform includes both position and rotation to point toward light direction
    gObjectWorld[OBJ_LIGHT_MARKER] = create_look_direction_matrix(gLight.position, gLight.direction);
    gObjectColor[OBJ_LIGHT_MARKER] = vec4(gLight.color, 0.0f);

    LOGF(eINFO, "  Light Marker: verts=%u indices=%u",
        gSimpleMesh[OBJ_LIGHT_MARKER].vertexCount, gSimpleMesh[OBJ_LIGHT_MARKER].indexCount);

    // Floor grid (multiple objects)
    LOGF(eINFO, "Generating Floor Grid starting at Object %u",
        OBJ_FLOOR_START);

    // Configure floor grid params
    uint32_t gridDimX = 1;          // tiles in X direction
    uint32_t gridDimZ = 1;          // tiles in Z direction
    float    tileSizeX = 1024;      // units wide
    float    tileSizeZ = 1024;      // units deep
    uint32_t segmentsPerTile = 4;   // quads per tile
    int      atlasSlot = 2;         // Use floor texture

    // Generate the floor grid (centered at origin)
    generate_multi_floor_grid(
        gridDimX, gridDimZ,
        tileSizeX, tileSizeZ,
        segmentsPerTile, atlasSlot);

    LOGF(eINFO, "Mesh generation complete:");
    LOGF(eINFO, "  Total objects: %u", NUM_NON_FLOOR_OBJECTS + gActiveFloorTiles);
    LOGF(eINFO, "  - Cloth: 1");
    LOGF(eINFO, "  - Sphere: 1");
    LOGF(eINFO, "  - Light Marker: 1");
    LOGF(eINFO, "  - Floor Tiles: %u (%ux%u grid)", gActiveFloorTiles, gFloorGridDimX, gFloorGridDimZ);
}

class Transformations: public IApp
{
public:
    bool Init()
    {
        // window and renderer setup
        RendererDesc settings;
        memset(&settings, 0, sizeof(settings));
        initGPUConfiguration(settings.pExtendedSettings);
        initRenderer(GetName(), &settings, &pRenderer);
        // check for init success
        if (!pRenderer)
        {
            ShowUnsupportedMessage(getUnsupportedGPUMsg());
            return false;
        }
        setupGPUConfigurationPlatformParameters(pRenderer, settings.pExtendedSettings);

        if (pRenderer->pGpu->mPipelineStatsQueries)
        {
            QueryPoolDesc poolDesc = {};
            poolDesc.mQueryCount = 3; // The count is 3 due to quest & multi-view use otherwise 2 is enough as we use 2 queries.
            poolDesc.mType = QUERY_TYPE_PIPELINE_STATISTICS;
            for (uint32_t i = 0; i < gDataBufferCount; ++i)
            {
                initQueryPool(pRenderer, &poolDesc, &pPipelineStatsQueryPool[i]);
            }
        }

        QueueDesc queueDesc = {};
        queueDesc.mType = QUEUE_TYPE_GRAPHICS;
        queueDesc.mFlag = QUEUE_FLAG_INIT_MICROPROFILE;
        initQueue(pRenderer, &queueDesc, &pGraphicsQueue);

        GpuCmdRingDesc cmdRingDesc = {};
        cmdRingDesc.pQueue = pGraphicsQueue;
        cmdRingDesc.mPoolCount = gDataBufferCount;
        cmdRingDesc.mCmdPerPoolCount = 1;
        cmdRingDesc.mAddSyncPrimitives = true;
        initGpuCmdRing(pRenderer, &cmdRingDesc, &gGraphicsCmdRing);

        for (uint32_t i = 0; i < gDataBufferCount; ++i)
        {
            initSemaphore(pRenderer, &pImageAcquiredSemaphore[i]);
        }

        initResourceLoaderInterface(pRenderer);

        RootSignatureDesc rootDesc = {};
        INIT_RS_DESC(rootDesc, "default.rootsig", "compute.rootsig");
        initRootSignature(pRenderer, &rootDesc);

        SamplerDesc samplerDesc = { FILTER_LINEAR,
                                    FILTER_LINEAR,
                                    MIPMAP_MODE_LINEAR,
                                    ADDRESS_MODE_CLAMP_TO_EDGE,
                                    ADDRESS_MODE_CLAMP_TO_EDGE,
                                    ADDRESS_MODE_CLAMP_TO_EDGE };
        addSampler(pRenderer, &samplerDesc, &pSkyBoxSampler);

        // Loads Skybox Textures
        for (int i = 0; i < 6; ++i)
        {
            TextureLoadDesc textureDesc = {};
            textureDesc.pFileName = pSkyBoxImageFileNames[i];
            textureDesc.ppTexture = &pSkyBoxTextures[i];
            // Textures representing color should be stored in SRGB or HDR format
            textureDesc.mCreationFlag = TEXTURE_CREATION_FLAG_SRGB;
            addResource(&textureDesc, NULL);
        }

        uint64_t       skyBoxDataSize = 4 * 6 * 6 * sizeof(float);
        BufferLoadDesc skyboxVbDesc = {};
        skyboxVbDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
        skyboxVbDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        skyboxVbDesc.mDesc.mSize = skyBoxDataSize;
        skyboxVbDesc.pData = gSkyBoxPoints;
        skyboxVbDesc.ppBuffer = &pSkyBoxVertexBuffer;
        addResource(&skyboxVbDesc, NULL);

        BufferLoadDesc ubDesc = {};
        ubDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ubDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        ubDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        ubDesc.pData = NULL;
        ubDesc.mDesc.mSize = sizeof(UniformBlock);

        for (uint32_t i = 0; i < gDataBufferCount; ++i)
        {
            for (uint32_t obj = 0; obj < gNumObjects; ++obj)
            {
                ubDesc.mDesc.pName = "UniformBuffer";
                ubDesc.ppBuffer = &pUniformBuffer[i][obj];
                addResource(&ubDesc, NULL);
            }
        }

        // Load fonts
        FontDesc font = {};
        font.pFontPath = "TitilliumText/TitilliumText-Bold.otf";
        fntDefineFonts(&font, 1, &gFontID);

        FontSystemDesc fontRenderDesc = {};
        fontRenderDesc.pRenderer = pRenderer;
        if (!initFontSystem(&fontRenderDesc))
            return false; // report?

        // Initialize Forge User Interface Rendering
        UserInterfaceDesc uiRenderDesc = {};
        uiRenderDesc.pRenderer = pRenderer;
        uiRenderDesc.pFontIds = &gFontID;
        uiRenderDesc.mFontIdsCount = 1;
        initUserInterface(&uiRenderDesc);

        // Initialize micro profiler and its UI.
        ProfilerDesc profiler = {};
        profiler.pRenderer = pRenderer;
        initProfiler(&profiler);

        // Gpu profiler can only be added after initProfile.
        gGpuProfileToken = initGpuProfiler(pRenderer, pGraphicsQueue, "Graphics");

        const uint32_t numScripts = TF_ARRAY_COUNT(gWindowTestScripts);
        LuaScriptDesc  scriptDescs[numScripts] = {};
        uint32_t       numScriptsFinal = numScripts;
        // For reload server test, use reload server test scripts
        if (!mSettings.mBenchmarking)
            numScriptsFinal = TF_ARRAY_COUNT(gReloadServerTestScripts);
        for (uint32_t i = 0; i < numScriptsFinal; ++i)
            scriptDescs[i].pScriptFileName = mSettings.mBenchmarking ? gWindowTestScripts[i] : gReloadServerTestScripts[i];
        DEFINE_LUA_SCRIPTS(scriptDescs, numScriptsFinal);

        waitForAllResourceLoads();

        // Camera Setup
        CameraMotionParameters cmp{ 160.0f, 200.0f, 200.0f };
        vec3                   camPos{ 0.0f, 50.0f, 105.0f };
        vec3                   lookAt{ vec3(0.0f, 10.0f, 0.0f) }; // Look at origin

        pCamera = initFpsCamera(camPos, lookAt);
        pCamera->setMotionParameters(cmp);

        LOGF(eINFO,
            "Camera initialized at (%.2f, %.2f, %.2f) looking at origin",
            camPos.x, camPos.y, camPos.z);

        // Initialize Object Transforms and Colors (before mesh generation)
        // Cloth
        gObjectWorld[OBJ_CLOTH] = mat4::identity();
        gObjectColor[OBJ_CLOTH] = vec4(1.0f, 1.0f, 1.0f, 1.0f); // White

        // Sphere
        gObjectWorld[OBJ_SPHERE] = mat4::translation(gSphereParams.position);
        gObjectColor[OBJ_SPHERE] = vec4(1.0f, 1.0f, 1.0f, 1.0f); // White

        // Light Marker
        gObjectWorld[OBJ_LIGHT_MARKER] = mat4::translation(gLight.position);
        gObjectColor[OBJ_LIGHT_MARKER] = vec4(gLight.color, 0.0f); // Self-lit (w=0)

        // Floor tiles
        for (uint32_t i = OBJ_FLOOR_START; i < OBJ_FLOOR_START + 64; ++i)
        {
            gObjectWorld[i] = mat4::identity();
            gObjectColor[i] = vec4(0.7f, 0.9f, 1.0f, 1.0f); // Light blue
        }

        LOGF(eINFO, "Object transforms initialized:");
        LOGF(eINFO, "  [%u] Cloth at (0, 61, 0)", OBJ_CLOTH);
        LOGF(eINFO, "  [%u] Sphere at (%.1f, %.1f, %.1f)",
            OBJ_SPHERE, gSphereParams.position.x, gSphereParams.position.y,
            gSphereParams.position.z);
        LOGF(eINFO, "  [%u] Light Marker at (%.1f, %.1f, %.1f)",
            OBJ_LIGHT_MARKER, gLight.position.x, gLight.position.y, gLight.position.z);
        LOGF(eINFO, "  [%u+] Floor tiles (will be generated)", OBJ_FLOOR_START);

        AddCustomInputBindings();
        initScreenshotCapturer(pRenderer, pGraphicsQueue, GetName());
        gFrameIndex = 0;

        return true;
    }

    void Exit()
    {
        exitScreenshotCapturer();

        exitCamera(pCamera);

        exitUserInterface();

        exitFontSystem();

        // Exit profile
        exitProfiler();

        for (uint32_t i = 0; i < gDataBufferCount; ++i)
        {
            for (uint32_t obj = 0; obj < gNumObjects; ++obj)
            {
                removeResource(pUniformBuffer[i][obj]);
            }
            if (pRenderer->pGpu->mPipelineStatsQueries)
            {
                exitQueryPool(pRenderer, pPipelineStatsQueryPool[i]);
            }
        }

        removeResource(pSkyBoxVertexBuffer);
        removeSampler(pRenderer, pSkyBoxSampler);

        for (uint i = 0; i < 6; ++i)
            removeResource(pSkyBoxTextures[i]);

        exitGpuCmdRing(pRenderer, &gGraphicsCmdRing);

        for (uint32_t i = 0; i < gDataBufferCount; ++i)
        {
            exitSemaphore(pRenderer, pImageAcquiredSemaphore[i]);
        }

        exitRootSignature(pRenderer);
        exitResourceLoaderInterface(pRenderer);

        exitQueue(pRenderer, pGraphicsQueue);

        exitRenderer(pRenderer);
        exitGPUConfiguration();
        pRenderer = NULL;
    }

    bool Load(ReloadDesc* pReloadDesc)
    {
        UNREF_PARAM(pReloadDesc);
        addShaders();
        addDescriptorSets();

        // we only need to reload gui when the size of window changed
        loadProfilerUI(mSettings.mWidth, mSettings.mHeight);

        gGuiWindowDesc = {};
        gGuiWindowDesc.mStartPos = vec2(mSettings.mWidth * 0.01f, mSettings.mHeight * 0.2f);
        gGuiWindowDesc.mStartSize = vec2(600.0f, 550.0f);
        gGuiWindowDesc.pWindowTitle = GetName();
        gGuiWindowDesc.mFlags =
            UI_WINDOW_TITLE | UI_WINDOW_SCALABLE | UI_WINDOW_MOVABLE | UI_WINDOW_MINIMIZABLE | UI_WINDOW_BORDER | UI_WINDOW_INIT_HEIGHT_FIT;

        if (!addSwapChain())
            return false;

        if (!addDepthBuffer())
            return false;

        generate_complex_mesh();
        addPipelines();

        prepareDescriptorSets();

        UserInterfaceLoadDesc uiLoad = {};
        uiLoad.mColorFormat = pSwapChain->ppRenderTargets[0]->mFormat;
        uiLoad.mHeight = mSettings.mHeight;
        uiLoad.mWidth = mSettings.mWidth;
        uiLoad.mVR2DLayer.mPosition = float3(gVR2DLayer.m2DLayerPosition.x, gVR2DLayer.m2DLayerPosition.y, gVR2DLayer.m2DLayerPosition.z);
        uiLoad.mVR2DLayer.mScale = gVR2DLayer.m2DLayerScale;
        loadUserInterface(&uiLoad);

        FontSystemLoadDesc fontLoad = {};
        fontLoad.mColorFormat = pSwapChain->ppRenderTargets[0]->mFormat;
        fontLoad.mHeight = mSettings.mHeight;
        fontLoad.mWidth = mSettings.mWidth;
        loadFontSystem(&fontLoad);

        return true;
    }

    void Unload(ReloadDesc* pReloadDesc)
    {
        UNREF_PARAM(pReloadDesc);
        waitQueueIdle(pGraphicsQueue);

        unloadFontSystem();
        unloadUserInterface();

        removePipelines();

        // Clean up all active objects (cloth, sphere, light marker, floor tiles)
        uint32_t totalActiveObjects = NUM_NON_FLOOR_OBJECTS + gActiveFloorTiles;

        LOGF(eINFO, "Cleaning up %u objects:", totalActiveObjects);
        LOGF(eINFO, "  - Non-floor objects: %u", NUM_NON_FLOOR_OBJECTS);
        LOGF(eINFO, "  - Floor tiles: %u", gActiveFloorTiles);

        for (uint32_t obj = 0; obj < totalActiveObjects; ++obj)
        {
            // Remove GPU buffers
            if (pMeshVertexBuffer[obj])
            {
                removeResource(pMeshVertexBuffer[obj]);
                pMeshVertexBuffer[obj] = NULL;
            }

            if (pMeshIndexBuffer[obj])
            {
                removeResource(pMeshIndexBuffer[obj]);
                pMeshIndexBuffer[obj] = NULL;
            }

            // Free CPU vertex data
            if (gMeshCpuMappedVertexData[obj])
            {
                tf_free(gMeshCpuMappedVertexData[obj]);
                gMeshCpuMappedVertexData[obj] = nullptr;
            }

            // Free SimpleMesh data
            free_simple_mesh(gSimpleMesh[obj]);

            // Free animation data edge arrays
            for (int i = 0; i < 4; ++i)
            {
                if (gAnimData[obj].edgeIndices[i])
                {
                    tf_free(gAnimData[obj].edgeIndices[i]);
                    gAnimData[obj].edgeIndices[i] = nullptr;
                }
            }
            gAnimData[obj].hasEdgeData = false;
        }

        // Reset floor tile tracking
        gActiveFloorTiles = 0;
        gFloorGridDimX = 0;
        gFloorGridDimZ = 0;

        LOGF(eINFO, "Object cleanup complete");

        removeSwapChain(pRenderer, pSwapChain);
        removeRenderTarget(pRenderer, pDepthBuffer);
        unloadProfilerUI();

        removeDescriptorSets();
        removeShaders();
    }

    void ConstrainCamera()
    {
        vec3 camPos = pCamera->getViewPosition();
        bool clamped = false;

        // Floor dimensions (from generate_complex_mesh)
        const float FLOOR_HALF_SIZE_X = 512.0f; // Total: 1024 units
        const float FLOOR_HALF_SIZE_Z = 512.0f; // Total: 1024 units
        const float EDGE_MARGIN = 10.0f;        // Stay 10 units from edge
        const float FLOOR_HEIGHT_OFFSET = 0.5f; // Stay 0.5 units above floor
        const float SKY_MAX = 200.f; // Maximum above the ground

        // Constrain X (left/right)
        float maxX = FLOOR_HALF_SIZE_X - EDGE_MARGIN; // 502
        float minX = -maxX;                           // -502
        if (camPos.x > maxX)
        {
            camPos.x = maxX;
            clamped = true;
        }
        else if (camPos.x < minX)
        {
            camPos.x = minX;
            clamped = true;
        }

        // Constrain Z (forward/back)
        float maxZ = FLOOR_HALF_SIZE_Z - EDGE_MARGIN; // 502
        float minZ = -maxZ;                           // -502
        if (camPos.z > maxZ)
        {
            camPos.z = maxZ;
            clamped = true;
        }
        else if (camPos.z < minZ)
        {
            camPos.z = minZ;
            clamped = true;
        }

        // Constrain Y (height - never go below floor)
        if (camPos.y < FLOOR_HEIGHT_OFFSET)
        {
            camPos.y = FLOOR_HEIGHT_OFFSET;
            clamped = true;
        }

        // let's not go too high either
        if (camPos.y > SKY_MAX)
        {
            camPos.y = SKY_MAX;
            clamped = true;
        }

        // Apply clamped position back to camera
        if (clamped)
        {
            pCamera->moveTo(camPos);
        }
    }

    void Update(float deltaTime)
    {
        // UI update
        if (gGuiActive)
        {
            if (UI_WINDOW_IS_VISIBLE(uiBeginWidgetWindow(&gGuiWindowDesc)))
            {
                float4 separatorColor = float4(0.3f, 0.3f, 0.3f, 1.0f);

                uiVerticalSeparator(separatorColor, 1.0f);

                if (pRenderer->pGpu->mPipelineStatsQueries)
                {
                    static float4 textColor = { 1.0f, 1.0f, 1.0f, 1.0f };

                    uiLayoutAutoTextRows(1);

                    uiLabel("Pipeline Stats", ALIGN_LEFT);

                    uiLayoutAutoTextRows(1);
                    uiDynamicText(&gPipelineStats, textColor, TEXT_MODE_WRAPPED, ALIGN_WRAPPED);

                    uiVerticalSeparator(separatorColor, 1.0f);
                }
            }
            uiEndWidgetWindow();
        }

        if (!uiIsFocused())
        {
            pCamera->onMove({ inputGetValue(0, CUSTOM_MOVE_X), inputGetValue(0, CUSTOM_MOVE_Y) });
            pCamera->onRotate({ inputGetValue(0, CUSTOM_LOOK_X), inputGetValue(0, CUSTOM_LOOK_Y) });
            pCamera->onMoveY(inputGetValue(0, CUSTOM_MOVE_UP));
            if (inputGetValue(0, CUSTOM_RESET_VIEW))
            {
                pCamera->resetView();
            }
            if (inputGetValue(0, CUSTOM_TOGGLE_FULLSCREEN))
            {
                toggleFullscreen(pWindow);
            }
            if (inputGetValue(0, CUSTOM_TOGGLE_UI))
            {
                gGuiActive = !gGuiActive;
            }
            if (inputGetValue(0, CUSTOM_DUMP_PROFILE))
            {
                dumpProfileData(GetName());
            }
            if (inputGetValue(0, CUSTOM_EXIT))
            {
                requestShutdown();
            }
        }

        pCamera->update(deltaTime);

        ConstrainCamera();

        // Log camera position every N frames
        {
            static uint32_t       logCounter = 0;
            static const uint32_t LOG_INTERVAL = 300;

            if (++logCounter >= LOG_INTERVAL)
            {
                vec3 camPos = pCamera->getViewPosition();
                LOGF(eINFO,
                    "Camera Position: (%.2f, %.2f, %.2f)", camPos.x, camPos.y, camPos.z);
                logCounter = 0;
            }
        }

        /************************************************************************/
        // Scene Update
        /************************************************************************/
        static float currentTime = 0.0f;
        currentTime += deltaTime * 1000.0f;

        // update camera with time
        CameraMatrix viewMat = pCamera->getViewMatrix();

        const float aspectInverse = (float)mSettings.mHeight / (float)mSettings.mWidth;
        const float horizontal_fov = PI / 2.0f;

        CameraMatrix projMat = camMatPerspectiveReverseZ(horizontal_fov, aspectInverse, 0.1f, 1000.0f);
        gUniformData.mProjectView = camMatMul(&projMat, &viewMat);

        // Light parameters
        gUniformData.mLightPosition = vec4(gLight.position, 0.0f);
        gUniformData.mLightColor = vec4(gLight.color * gLight.intensity, 1.0f);
        gUniformData.mCameraPosition = vec4(pCamera->getViewPosition(), 0.0f);
        gUniformData.mLightDirection = vec4(gLight.direction, 0.0f);

        // Skybox view (fixed at origin)
        viewMat = camMatSetTranslation(&viewMat, vec3(0));
        gUniformData.mSkyProjectView = camMatMul(&projMat, &viewMat);

        TuneLights(deltaTime);
        MorphObjects(deltaTime);
    }

    void TuneLights(float deltaTime)
    {

        const float rotSpeed = 50.0f * deltaTime;    // degrees per second

        if (GetAsyncKeyState(VK_OEM_COMMA) & 0x8000)
        {
            // Rotate light direction around X axis (counter-clockwise)
            float angle = rotSpeed * (3.14159f / 180.0f); // convert to radians
            float newY = gLight.direction.y * cosf(angle) - gLight.direction.z * sinf(angle);
            float newZ = gLight.direction.y * sinf(angle) + gLight.direction.z * cosf(angle);
            gLight.direction.y = newY;
            gLight.direction.z = newZ;

            // Normalize
            float len = sqrtf(gLight.direction.x * gLight.direction.x + gLight.direction.y * gLight.direction.y +
                              gLight.direction.z * gLight.direction.z);
            gLight.direction.x /= len;
            gLight.direction.y /= len;
            gLight.direction.z /= len;

            LOGF(eINFO, "Light Dir: (%.2f, %.2f, %.2f)", gLight.direction.x, gLight.direction.y, gLight.direction.z);
        }

        if (GetAsyncKeyState(VK_OEM_PERIOD) & 0x8000)
        {
            // Rotate light direction around X axis (clockwise)
            float angle = -rotSpeed * (3.14159f / 180.0f); // negative for opposite direction
            float newY = gLight.direction.y * cosf(angle) - gLight.direction.z * sinf(angle);
            float newZ = gLight.direction.y * sinf(angle) + gLight.direction.z * cosf(angle);
            gLight.direction.y = newY;
            gLight.direction.z = newZ;

            // Normalize
            float len = sqrtf(gLight.direction.x * gLight.direction.x + gLight.direction.y * gLight.direction.y +
                              gLight.direction.z * gLight.direction.z);
            gLight.direction.x /= len;
            gLight.direction.y /= len;
            gLight.direction.z /= len;

            LOGF(eINFO, "Light Dir: (%.2f, %.2f, %.2f)", gLight.direction.x, gLight.direction.y, gLight.direction.z);
        }
    }

    void MorphObjects(float deltaTime)
    {
        gUniformData.mGeometryWeight[0] = 1.0f;

        // Cloth physics
        const uint32_t kClothIndex = (uint32_t)OBJ_CLOTH;

        if (gMeshCpuMappedVertexData[kClothIndex] && gCloth.GetParticleCount() > 0)
        {
            const float zNudge = 50.f;

            // Keyboard controls for cloth manipulation
            if (GetAsyncKeyState('I') & 0x8000)
            {
                gCloth.NudgeEdge(CLOTH_EDGE_TOP, zNudge * deltaTime);
            }
            if (GetAsyncKeyState('K') & 0x8000)
            {
                gCloth.NudgeEdge(CLOTH_EDGE_TOP, -zNudge * deltaTime);
            }

            // Keyboard controls for sphere movement
            if (GetAsyncKeyState('O') & 0x8000)
            {
                gSphereParams.position.z += zNudge / 2 * deltaTime;
                gObjectWorld[OBJ_SPHERE] = mat4::translation(gSphereParams.position);
                gCloth.SetSphereCollider(gSphereParams.position, gSphereParams.radius);
            }
            if (GetAsyncKeyState('L') & 0x8000)
            {
                gSphereParams.position.z -= zNudge / 2 * deltaTime;
                gObjectWorld[OBJ_SPHERE] = mat4::translation(gSphereParams.position);
                gCloth.SetSphereCollider(gSphereParams.position, gSphereParams.radius);
            }

            // Release cloth pins
            if (GetAsyncKeyState('R') & 0x8000)
            {
                gCloth.ReleasePinCloth();
            }

            // Update cloth physics
            gCloth.Update(deltaTime);

            // Copy simulated positions back to CPU vertex data
            uint8_t* vertexData = (uint8_t*)gMeshCpuMappedVertexData[kClothIndex];
            uint32_t stride = gMeshVertexLayout.mBindings[0].mStride; // 52 bytes

            vec3* positions = gCloth.GetCurrentPositions();
            for (uint32_t i = 0; i < gCloth.GetParticleCount(); i++)
            {
                float* pos = (float*)(vertexData + i * stride);
                pos[0] = positions[i].x;
                pos[1] = positions[i].y;
                pos[2] = positions[i].z;
            }

            // Copy normals (offset 16 in vertex buffer)
            vec3* normals = gCloth.GetVertexNormals();
            for (uint32_t i = 0; i < gCloth.GetParticleCount(); i++)
            {
                float* norm = (float*)(vertexData + i * stride + 16);
                norm[0] = normals[i].x;
                norm[1] = normals[i].y;
                norm[2] = normals[i].z;
            }

            // Upload modified vertices to GPU
            BufferUpdateDesc vbUpdate = { pMeshVertexBuffer[kClothIndex] };
            beginUpdateResource(&vbUpdate);
            memcpy(vbUpdate.pMappedData,
                gMeshCpuMappedVertexData[kClothIndex],
                gMeshCpuMappedVertexSize[kClothIndex]);
            endUpdateResource(&vbUpdate);
        }
    }

    void Draw()
    {
        if ((bool)pSwapChain->mEnableVsync != mSettings.mVSyncEnabled)
        {
            waitQueueIdle(pGraphicsQueue);
            ::toggleVSync(pRenderer, &pSwapChain);
        }

        uint32_t swapchainImageIndex;
        acquireNextImage(pRenderer, pSwapChain, pImageAcquiredSemaphore[gFrameIndex], NULL, &swapchainImageIndex);

        RenderTarget*     pRenderTarget = pSwapChain->ppRenderTargets[swapchainImageIndex];
        GpuCmdRingElement elem = getNextGpuCmdRingElement(&gGraphicsCmdRing, true, 1);

        // Stall if CPU is running "gDataBufferCount" frames ahead of GPU
        FenceStatus fenceStatus;
        getFenceStatus(pRenderer, elem.pFence, &fenceStatus);
        if (fenceStatus == FENCE_STATUS_INCOMPLETE)
            waitForFences(pRenderer, 1, &elem.pFence);

        // Reset cmd pool for this frame
        resetCmdPool(pRenderer, elem.pCmdPool);

        if (pRenderer->pGpu->mPipelineStatsQueries)
        {
            QueryData data3D = {};
            QueryData data2D = {};
            getQueryData(pRenderer, pPipelineStatsQueryPool[gFrameIndex], 0, &data3D);
            getQueryData(pRenderer, pPipelineStatsQueryPool[gFrameIndex], 1, &data2D);
            bformat(&gPipelineStats,
                    "  Pipeline Stats 3D:\n"
                    "      VS invocations:      %u\n"
                    "      PS invocations:      %u\n"
                    "      Clipper invocations: %u\n"
                    "      IA primitives:       %u\n"
                    "      Clipper primitives:  %u\n"
                    "\n"
                    "  Pipeline Stats 2D UI:\n"
                    "      VS invocations:      %u\n"
                    "      PS invocations:      %u\n"
                    "      Clipper invocations: %u\n"
                    "      IA primitives:       %u\n"
                    "      Clipper primitives:  %u\n",
                    data3D.mPipelineStats.mVSInvocations, data3D.mPipelineStats.mPSInvocations, data3D.mPipelineStats.mCInvocations,
                    data3D.mPipelineStats.mIAPrimitives, data3D.mPipelineStats.mCPrimitives, data2D.mPipelineStats.mVSInvocations,
                    data2D.mPipelineStats.mPSInvocations, data2D.mPipelineStats.mCInvocations, data2D.mPipelineStats.mIAPrimitives,
                    data2D.mPipelineStats.mCPrimitives);
        }

        Cmd* cmd = elem.pCmds[0];
        beginCmd(cmd);
        cmdBindDescriptorSet(cmd, 0, pDescriptorSetPersistent);
        cmdBindDescriptorSet(cmd, gFrameIndex, pDescriptorSetPerFrame);

        cmdBeginGpuFrameProfile(cmd, gGpuProfileToken);
        if (pRenderer->pGpu->mPipelineStatsQueries)
        {
            cmdResetQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], 0, 2);
            QueryDesc queryDesc = { 0 };
            cmdBeginQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], &queryDesc);
        }

        RenderTargetBarrier barriers[] = {
            { pRenderTarget, RESOURCE_STATE_PRESENT, RESOURCE_STATE_RENDER_TARGET },
        };
        cmdResourceBarrier(cmd, 0, NULL, 0, NULL, 1, barriers);

        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Draw Scene");

        // simply record the screen cleaning command
        BindRenderTargetsDesc bindRenderTargets = {};
        bindRenderTargets.mRenderTargetCount = 1;
        bindRenderTargets.mRenderTargets[0] = { pRenderTarget, LOAD_ACTION_CLEAR };
        bindRenderTargets.mDepthStencil = { pDepthBuffer, LOAD_ACTION_CLEAR };
        cmdBindRenderTargets(cmd, &bindRenderTargets);
        cmdSetViewport(cmd, 0.0f, 0.0f, (float)pRenderTarget->mWidth, (float)pRenderTarget->mHeight, 0.0f, 1.0f);
        cmdSetScissor(cmd, 0, 0, pRenderTarget->mWidth, pRenderTarget->mHeight);

        const uint32_t skyboxVbStride = sizeof(float) * 4;
        // draw skybox
        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Draw Skybox");
        cmdSetViewport(cmd, 0.0f, 0.0f, (float)pRenderTarget->mWidth, (float)pRenderTarget->mHeight, 1.0f, 1.0f);
        cmdBindPipeline(cmd, pSkyBoxDrawPipeline);
        cmdBindVertexBuffer(cmd, 1, &pSkyBoxVertexBuffer, &skyboxVbStride, NULL);
        cmdDraw(cmd, 36, 0);
        cmdSetViewport(cmd, 0.0f, 0.0f, (float)pRenderTarget->mWidth, (float)pRenderTarget->mHeight, 0.0f, 1.0f);
        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);

        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Draw Objects");
        cmdBindPipeline(cmd, pMeshPipeline);

        // Calculate total number of active objects to draw
        uint32_t totalActiveObjects = NUM_NON_FLOOR_OBJECTS + gActiveFloorTiles;

        // Draw all active objects with independent uniforms
        for (uint32_t obj = 0; obj < totalActiveObjects; ++obj)
        {
            if (obj == OBJ_LIGHT_MARKER)
            {
                // Light marker: update both position and rotation every frame
                gUniformData.mToWorldMat = create_look_direction_matrix(gLight.position, gLight.direction);
                gUniformData.mColor = vec4(gLight.color, 0.0f); // w=0, self-lit shader path
            }
            else
            {
                gUniformData.mToWorldMat = gObjectWorld[obj];
                gUniformData.mColor = gObjectColor[obj];
            }

            gUniformData.mGeometryWeight[0] = 1.0f;

            BufferUpdateDesc objUb = { pUniformBuffer[gFrameIndex][obj] };
            beginUpdateResource(&objUb);
            memcpy(objUb.pMappedData, &gUniformData, sizeof(gUniformData));
            endUpdateResource(&objUb);

            uint32_t setIndex = gFrameIndex * gNumObjects + obj;
            cmdBindDescriptorSet(cmd, setIndex, pDescriptorSetPerFrame);

            // --- Bind THIS object's vertex + index buffers ---
            cmdBindVertexBuffer(cmd, 1, &pMeshVertexBuffer[obj], &gMeshVertexLayout.mBindings[0].mStride, nullptr);
            cmdBindIndexBuffer(cmd, pMeshIndexBuffer[obj], INDEX_TYPE_UINT16, 0);

            // --- Draw ---
            cmdDrawIndexed(cmd, gMeshIndexCount[obj], 0, 0);
        }

        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);

        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken); // Draw Scene
        cmdBindRenderTargets(cmd, NULL);

        if (pRenderer->pGpu->mPipelineStatsQueries)
        {
            QueryDesc queryDesc = { 0 };
            cmdEndQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], &queryDesc);

            queryDesc = { 1 };
            cmdBeginQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], &queryDesc);
        }

        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Draw UI");

        RenderTarget* uiRt = cmdBeginDrawingUserInterface(cmd, pSwapChain, pRenderTarget);
        {
            gFrameTimeDraw.mFontColor = 0xff00ffff;
            gFrameTimeDraw.mFontSize = 18.0f;
            gFrameTimeDraw.mFontID = gFontID;
            float2 txtSizePx = cmdDrawCpuProfile(cmd, float2(8.f, 15.f), &gFrameTimeDraw);
            cmdDrawGpuProfile(cmd, float2(8.f, txtSizePx.y + 75.f), gGpuProfileToken, &gFrameTimeDraw);

            cmdDrawUserInterface(cmd, uiRt, gGpuProfileToken);
        }
        cmdEndDrawingUserInterface(cmd, pSwapChain);

        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);

        barriers[0] = { pRenderTarget, RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_PRESENT };
        cmdResourceBarrier(cmd, 0, NULL, 0, NULL, 1, barriers);

        cmdEndGpuFrameProfile(cmd, gGpuProfileToken);

        if (pRenderer->pGpu->mPipelineStatsQueries)
        {
            QueryDesc queryDesc = { 1 };
            cmdEndQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], &queryDesc);
            cmdResolveQuery(cmd, pPipelineStatsQueryPool[gFrameIndex], 0, 2);
        }

        endCmd(cmd);

        FlushResourceUpdateDesc flushUpdateDesc = {};
        flushUpdateDesc.mNodeIndex = 0;
        flushResourceUpdates(&flushUpdateDesc);
        Semaphore* waitSemaphores[2] = { flushUpdateDesc.pOutSubmittedSemaphore, pImageAcquiredSemaphore[gFrameIndex] };

        QueueSubmitDesc submitDesc = {};
        submitDesc.mCmdCount = 1;
        submitDesc.mSignalSemaphoreCount = 1;
        submitDesc.mWaitSemaphoreCount = TF_ARRAY_COUNT(waitSemaphores);
        submitDesc.ppCmds = &cmd;
        submitDesc.ppSignalSemaphores = &elem.pSemaphore;
        submitDesc.ppWaitSemaphores = waitSemaphores;
        submitDesc.pSignalFence = elem.pFence;
        queueSubmit(pGraphicsQueue, &submitDesc);

        QueuePresentDesc presentDesc = {};
        presentDesc.mIndex = (uint8_t)swapchainImageIndex;
        presentDesc.mWaitSemaphoreCount = 1;
        presentDesc.pSwapChain = pSwapChain;
        presentDesc.ppWaitSemaphores = &elem.pSemaphore;
        presentDesc.mSubmitDone = true;

        queuePresent(pGraphicsQueue, &presentDesc);
        flipProfiler();

        gFrameIndex = (gFrameIndex + 1) % gDataBufferCount;
    }

    const char* GetName() { return "Cloth Demo (Gustavo Oliveira)"; }

    bool addSwapChain()
    {
        SwapChainDesc swapChainDesc = {};
        swapChainDesc.mWindowHandle = pWindow->handle;
        swapChainDesc.mPresentQueueCount = 1;
        swapChainDesc.ppPresentQueues = &pGraphicsQueue;
        swapChainDesc.mWidth = mSettings.mWidth;
        swapChainDesc.mHeight = mSettings.mHeight;
        swapChainDesc.mImageCount = getRecommendedSwapchainImageCount(pRenderer, &pWindow->handle);
        swapChainDesc.mColorFormat = getSupportedSwapchainFormat(pRenderer, &swapChainDesc, COLOR_SPACE_SDR_SRGB);
        swapChainDesc.mColorSpace = COLOR_SPACE_SDR_SRGB;
        swapChainDesc.mEnableVsync = mSettings.mVSyncEnabled;
        swapChainDesc.mFlags = SWAP_CHAIN_CREATION_FLAG_ENABLE_2D_VR_LAYER;
        swapChainDesc.mVR.m2DLayer = gVR2DLayer;

        ::addSwapChain(pRenderer, &swapChainDesc, &pSwapChain);

        return pSwapChain != NULL;
    }

    bool addDepthBuffer()
    {
        // Add depth buffer
        ESRAM_BEGIN_ALLOC(pRenderer, "Depth", 0);

        RenderTargetDesc depthRT = {};
        depthRT.mArraySize = 1;
        depthRT.mClearValue.depth = 0.0f;
        depthRT.mClearValue.stencil = 0;
        depthRT.mDepth = 1;
        depthRT.mFormat = TinyImageFormat_D32_SFLOAT;
        depthRT.mStartState = RESOURCE_STATE_DEPTH_WRITE;
        depthRT.mHeight = mSettings.mHeight;
        depthRT.mSampleCount = SAMPLE_COUNT_1;
        depthRT.mSampleQuality = 0;
        depthRT.mWidth = mSettings.mWidth;
        depthRT.mFlags = TEXTURE_CREATION_FLAG_ESRAM | TEXTURE_CREATION_FLAG_ON_TILE | TEXTURE_CREATION_FLAG_VR_MULTIVIEW;
        addRenderTarget(pRenderer, &depthRT, &pDepthBuffer);

        ESRAM_END_ALLOC(pRenderer);

        return pDepthBuffer != NULL;
    }

    void addDescriptorSets()
    {
        DescriptorSetDesc descPersisent = SRT_SET_DESC(SrtData, Persistent, 1, 0);
        addDescriptorSet(pRenderer, &descPersisent, &pDescriptorSetPersistent);
        DescriptorSetDesc descUniforms = SRT_SET_DESC(SrtData, PerFrame, gDataBufferCount * gNumObjects, 0);
        addDescriptorSet(pRenderer, &descUniforms, &pDescriptorSetPerFrame);
    }

    void removeDescriptorSets()
    {
        removeDescriptorSet(pRenderer, pDescriptorSetPerFrame);
        removeDescriptorSet(pRenderer, pDescriptorSetPersistent);
    }

    void addShaders()
    {
        ShaderLoadDesc skyShader = {};
        skyShader.mVert.pFileName = "skybox.vert";
        skyShader.mFrag.pFileName = "skybox.frag";

        ShaderLoadDesc basicShader = {};
        basicShader.mVert.pFileName = "basic.vert";
        basicShader.mFrag.pFileName = "basic.frag";

        addShader(pRenderer, &skyShader, &pSkyBoxDrawShader);
        addShader(pRenderer, &basicShader, &pSphereShader);
    }

    void removeShaders()
    {
        removeShader(pRenderer, pSphereShader);
        removeShader(pRenderer, pSkyBoxDrawShader);
    }

    void addPipelines()
    {
        RasterizerStateDesc rasterizerStateDesc = {};
        rasterizerStateDesc.mCullMode = CULL_MODE_NONE;

        RasterizerStateDesc sphereRasterizerStateDesc = {};
        // sphereRasterizerStateDesc.mCullMode = CULL_MODE_FRONT;
        //sphereRasterizerStateDesc.mFillMode = FILL_MODE_WIREFRAME;
        sphereRasterizerStateDesc.mCullMode = CULL_MODE_NONE;

        DepthStateDesc depthStateDesc = {};
        depthStateDesc.mDepthTest = true;
        depthStateDesc.mDepthWrite = true;

        depthStateDesc.mDepthFunc = CMP_GEQUAL;

        PipelineDesc desc = {};
        desc.mType = PIPELINE_TYPE_GRAPHICS;
        PIPELINE_LAYOUT_DESC(desc, SRT_LAYOUT_DESC(SrtData, Persistent), SRT_LAYOUT_DESC(SrtData, PerFrame), NULL, NULL);
        GraphicsPipelineDesc& pipelineSettings = desc.mGraphicsDesc;
        pipelineSettings.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
        pipelineSettings.mRenderTargetCount = 1;
        pipelineSettings.pDepthState = &depthStateDesc;
        pipelineSettings.pColorFormats = &pSwapChain->ppRenderTargets[0]->mFormat;
        pipelineSettings.mSampleCount = pSwapChain->ppRenderTargets[0]->mSampleCount;
        pipelineSettings.mSampleQuality = pSwapChain->ppRenderTargets[0]->mSampleQuality;
        pipelineSettings.mDepthStencilFormat = pDepthBuffer->mFormat;
        pipelineSettings.pShaderProgram = pSphereShader;
        pipelineSettings.pVertexLayout = &gMeshVertexLayout;
        pipelineSettings.pRasterizerState = &sphereRasterizerStateDesc;
        addPipeline(pRenderer, &desc, &pMeshPipeline);

        // layout and pipeline for skybox draw
        VertexLayout vertexLayout = {};
        vertexLayout.mBindingCount = 1;
        vertexLayout.mBindings[0].mStride = sizeof(float4);
        vertexLayout.mAttribCount = 1;
        vertexLayout.mAttribs[0].mSemantic = SEMANTIC_POSITION;
        vertexLayout.mAttribs[0].mFormat = TinyImageFormat_R32G32B32A32_SFLOAT;
        vertexLayout.mAttribs[0].mBinding = 0;
        vertexLayout.mAttribs[0].mLocation = 0;
        vertexLayout.mAttribs[0].mOffset = 0;
        pipelineSettings.pVertexLayout = &vertexLayout;

        pipelineSettings.pDepthState = NULL;
        pipelineSettings.pRasterizerState = &rasterizerStateDesc;
        pipelineSettings.pShaderProgram = pSkyBoxDrawShader; //-V519
        addPipeline(pRenderer, &desc, &pSkyBoxDrawPipeline);
    }

    void removePipelines()
    {
        removePipeline(pRenderer, pSkyBoxDrawPipeline);
        removePipeline(pRenderer, pMeshPipeline);
    }

    void prepareDescriptorSets()
    {
        // Prepare descriptor sets
        DescriptorData params[7] = {};
        params[0].mIndex = SRT_RES_IDX(SrtData, Persistent, gRightTexture);
        params[0].ppTextures = &pSkyBoxTextures[0];
        params[1].mIndex = SRT_RES_IDX(SrtData, Persistent, gLeftTexture);
        params[1].ppTextures = &pSkyBoxTextures[1];
        params[2].mIndex = SRT_RES_IDX(SrtData, Persistent, gTopTexture);
        params[2].ppTextures = &pSkyBoxTextures[2];
        params[3].mIndex = SRT_RES_IDX(SrtData, Persistent, gBotTexture);
        params[3].ppTextures = &pSkyBoxTextures[3];
        params[4].mIndex = SRT_RES_IDX(SrtData, Persistent, gFrontTexture);
        params[4].ppTextures = &pSkyBoxTextures[4];
        params[5].mIndex = SRT_RES_IDX(SrtData, Persistent, gBackTexture);
        params[5].ppTextures = &pSkyBoxTextures[5];
        params[6].mIndex = SRT_RES_IDX(SrtData, Persistent, gSampler);
        params[6].ppSamplers = &pSkyBoxSampler;
        updateDescriptorSet(pRenderer, 0, pDescriptorSetPersistent, TF_ARRAY_COUNT(params), params);

        for (uint32_t i = 0; i < gDataBufferCount; ++i)
        {
            for (uint32_t obj = 0; obj < gNumObjects; ++obj)
            {
                DescriptorData uParams[1] = {};
                uParams[0].mIndex = SRT_RES_IDX(SrtData, PerFrame, gUniformBlock);
                uParams[0].ppBuffers = &pUniformBuffer[i][obj];
                uint32_t setIndex = i * gNumObjects + obj;
                updateDescriptorSet(pRenderer, setIndex, pDescriptorSetPerFrame, 1, uParams);
            }
        }
    }
};
DEFINE_APPLICATION_MAIN(Transformations)
