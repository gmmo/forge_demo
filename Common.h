#pragma once

//////////////////////////////////////////////////////////////////////////////////////////////////
//                                      Enums
//////////////////////////////////////////////////////////////////////////////////////////////////

// Object layout enum - defines the index for each type of object
enum ObjectIndex
{
    OBJ_CLOTH = 0,        // Hanging cloth simulation
    OBJ_SPHERE = 1,       // Physics sphere
    OBJ_LIGHT_MARKER = 2, // Visual marker for light position
    OBJ_FLOOR_START = 3   // Floor meshes start here
};

enum PlaneOrientation
{
    PLANE_XZ = 0, // Floor/Ground (normal points up +Y)
    PLANE_XY = 1, // Wall/Hanging Cloth (normal points forward +Z)
    PLANE_YZ = 2  // Side Wall (normal points right +X)
};


//////////////////////////////////////////////////////////////////////////////////////////////////
//                                      Structs
//////////////////////////////////////////////////////////////////////////////////////////////////

struct SimpleMesh
{
    float*      verts;
    float*      normals;
    uint8_t*    colors;
    float*      UVs;

    uint16_t*   indices;

    uint32_t    vertexCount;
    uint32_t    indexCount;

    // Edge tracking: [0]=top, [1]=bottom, [2]=left, [3]=right
    uint16_t*   edgeIndices[4];
    uint32_t    edgeCounts[4]; // Number of vertices in each edge
    uint16_t    centerIndex;   // Index of vertex closest to center
};

// Sphere parameters — single source of truth for size, position, mesh detail
struct SphereParams
{
    vec3     position = { 0.0f, 12.0f, 10.0f };
    float    radius = 12.0f;
    uint32_t segments = 24;
    uint32_t rings = 12;
};

// Edge and center tracking for animation (copied from SimpleMesh after upload)
struct MeshAnimationData
{
    uint16_t* edgeIndices[4]; // Pointers to edge vertex indices
    uint32_t  edgeCounts[4];  // Number of vertices in each edge
    uint16_t  centerIndex;    // Center vertex index
    bool      hasEdgeData;    // Whether this mesh has edge data
};

//////////////////////////////////////////////////////////////////////////////////////////////////
//                                      Constants/Defines
//////////////////////////////////////////////////////////////////////////////////////////////////

// Floor grid configuration
static const uint32_t MAX_FLOOR_GRID_DIM = 32;                                   // Max 32x32 grid
static const uint32_t MAX_FLOOR_TILES = MAX_FLOOR_GRID_DIM * MAX_FLOOR_GRID_DIM; // 1024
static const uint32_t NUM_NON_FLOOR_OBJECTS = 3;                                 // Cloth + Sphere + Light

// Update gNumObjects to accommodate maximum possible objects
static const uint32_t gNumObjects = NUM_NON_FLOOR_OBJECTS + MAX_FLOOR_TILES;

//////////////////////////////////////////////////////////////////////////////////////////////////
//                                      Globals
//////////////////////////////////////////////////////////////////////////////////////////////////

// Look up for indices in the mesh
static MeshAnimationData    gAnimData[gNumObjects] = {};

// Buffers for the mesh
SimpleMesh                  gSimpleMesh[gNumObjects] = {};

// per-object transforms/colors (two planes)
static mat4                 gObjectWorld[gNumObjects];
static vec4                 gObjectColor[gNumObjects];

static  SphereParams        gSphereParams;
