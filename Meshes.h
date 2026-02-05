#pragma once

#include "Common.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
// MESH GENERATION MODULE - Procedural geometry creation with texture atlas support
// - Generates parametric meshes (planes, spheres, arrows) with configurable tessellation
// - Maps all geometry UVs into a shared 2x2 texture atlas (marble, cloth, floor, checkerboard)
//
// CORE MESH GENERATORS:
// • generate_plane_mesh() - Creates grid mesh (XZ/XY/YZ orientation) with edge tracking for cloth simulation
// • generate_sphere_mesh() - Creates UV sphere with configurable segments/rings for collision visualization
// • generate_arrow_mesh() - Creates low-poly arrow (cone tip + cylinder shaft + cap) for debug visualization
// • generate_multi_floor_grid() - Creates centered grid of floor tiles with automatic positioning
//
// TEXTURE ATLAS MAPPING:
// • remap_uv_to_atlas() - Maps local UV (0..1) into atlas slot (0=marble, 1=cloth, 2=floor, 3=checkerboard)
// • Each atlas tile occupies 0.5x0.5 UV space with inset bias to prevent texture bleeding
//
// GPU UPLOAD & ANIMATION:
// • upload_mesh_to_gpu() - Uploads vertex/index buffers to GPU with CPU-side mapping for dynamic updates
// • copy_animation_data() - Extracts edge/center vertex indices for physics simulation
//////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////
//                                      Structs
//////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////
// Texture Atlas UV Lookup Table
//   Layout in the atlas:
//     0    marble          0.00    0.00
//     1    cloth           0.50    0.00
//     2    floot           0.00    0.50
//     3    checkerboard    0.50    0.50
//////////////////////////////////////////

struct UVTableLookup
{
    float texture0UVRange[2]; // marble:        top-left     (0.00, 0.00)
    float texture1UVRange[2]; // cloth:         top-right    (0.50, 0.00)
    float texture2UVRange[2]; // floor:         bottom-left  (0.00, 0.50)
    float texture3UVRange[2]; // checkerboard:  bottom-right (0.50, 0.50)
};

//////////////////////////////////////////////////////////////////////////////////////////////////
//                                      Globals
//////////////////////////////////////////////////////////////////////////////////////////////////

// Track actual number of floor tiles being used
static uint32_t gActiveFloorTiles = 0;
static uint32_t gFloorGridDimX = 0;
static uint32_t gFloorGridDimZ = 0;

Buffer*  pMeshVertexBuffer[gNumObjects] = { NULL };
Buffer*  pMeshIndexBuffer[gNumObjects] = { NULL };
uint32_t gMeshIndexCount[gNumObjects] = { 0 };

Pipeline*    pMeshPipeline = NULL;
VertexLayout gMeshVertexLayout = {};
uint32_t     gMeshLayoutType = 0;

static void*  gMeshCpuMappedVertexData[gNumObjects] = {};
static size_t gMeshCpuMappedVertexSize[gNumObjects] = {};

static UVTableLookup gAtlasUVs = {
    { 0.00f, 0.00f },
    { 0.50f, 0.00f },
    { 0.00f, 0.50f },
    { 0.50f, 0.50f },
};

// Tile size in UV space (each tile is half the atlas on each axis)
static const float ATLAS_TILE_UV = 0.5f;

//////////////////////////////////////////////////////////////////////////////////////////////////
//                                      Forward Declarations/Prototypes
//////////////////////////////////////////////////////////////////////////////////////////////////

static void generate_plane_mesh(
    SimpleMesh& mesh, float width, float depth, uint32_t xSegments,
    uint32_t zSegments, int atlasSlot,
    PlaneOrientation orientation, vec3 worldCenter, float uvTiling);

static void upload_mesh_to_gpu(
    const SimpleMesh& mesh,
    VertexLayout& layout,
    Buffer** ppVertexBuffer,
    Buffer** ppIndexBuffer, uint32_t& outIndexCount, void** ppCpuVertexData,
    size_t& outCpuVertexSize);

static void copy_animation_data(
    uint32_t objIndex,
    const SimpleMesh& mesh);


// --------------------------------------------------------------------------
// Helper: remap a local UV (0..1, 0..1) into atlas space for a given slot.
//   slot   – index into gAtlasUVs (0-3)
//   localU – per-object U in [0..1]
//   localV – per-object V in [0..1]
//   outU   – atlas U written here
//   outV   – atlas V written here
// --------------------------------------------------------------------------
static inline void remap_uv_to_atlas(
    int slot,
    float localU, float localV,
    float* outU, float* outV)
{
    // Add bias to prevent bleeding, however it does not work well with filtering
    const float UV_INSET = 0.004f;
    const float* base = ((const float*)&gAtlasUVs) + slot * 2; // each entry is 2 floats
    const float effectiveTileSize = ATLAS_TILE_UV - 2.0f * UV_INSET;

    *outU = base[0] + UV_INSET + localU * effectiveTileSize;
    *outV = base[1] + UV_INSET + localV * effectiveTileSize;
}


// Release all memory from a mesh
static void free_simple_mesh(SimpleMesh& mesh)
{
    if (mesh.verts)
    {
        tf_free(mesh.verts);
        mesh.verts = nullptr;
    }
    if (mesh.normals)
    {
        tf_free(mesh.normals);
        mesh.normals = nullptr;
    }
    if (mesh.colors)
    {
        tf_free(mesh.colors);
        mesh.colors = nullptr;
    }
    if (mesh.UVs)
    {
        tf_free(mesh.UVs);
        mesh.UVs = nullptr;
    }
    if (mesh.indices)
    {
        tf_free(mesh.indices);
        mesh.indices = nullptr;
    }

    // Free edge arrays
    for (int i = 0; i < 4; ++i)
    {
        if (mesh.edgeIndices[i])
        {
            tf_free(mesh.edgeIndices[i]);
            mesh.edgeIndices[i] = nullptr;
        }
        mesh.edgeCounts[i] = 0;
    }

    mesh.vertexCount = 0;
    mesh.indexCount = 0;
    mesh.centerIndex = 0;
}


// Creates a grid of floor tiles centered at world origin (0, 0, 0)
static void generate_multi_floor_grid(
    uint32_t gridDimX, uint32_t gridDimZ,
    float tileSizeX, float tileSizeZ,
    uint32_t segmentsPerTile,
    int atlasSlot)
{
    // Safety checks
    if (gridDimX == 0 || gridDimZ == 0)
    {
        LOGF(eWARNING, "generate_multi_floor_grid: Grid dimensions cannot be zero");
        return;
    }

    if (gridDimX > MAX_FLOOR_GRID_DIM || gridDimZ > MAX_FLOOR_GRID_DIM)
    {
        LOGF(eWARNING,
                "generate_multi_floor_grid: Grid dimensions %ux%u exceed maximum %ux%u",
                gridDimX, gridDimZ, MAX_FLOOR_GRID_DIM,
                MAX_FLOOR_GRID_DIM);
        return;
    }

    // Store grid dimensions
    gFloorGridDimX = gridDimX;
    gFloorGridDimZ = gridDimZ;
    gActiveFloorTiles = gridDimX * gridDimZ;

    // Calculate total grid size in world space
    float totalGridSizeX = tileSizeX * gridDimX;
    float totalGridSizeZ = tileSizeZ * gridDimZ;

    // Calculate offset to center the grid at origin (0, 0, 0)
    float gridStartX = -totalGridSizeX * 0.5f;
    float gridStartZ = -totalGridSizeZ * 0.5f;

    LOGF(eINFO, "Generating %ux%u floor grid (%u tiles total)",
        gridDimX, gridDimZ, gActiveFloorTiles);
    LOGF(eINFO, "  Tile size: %.2f x %.2f", tileSizeX, tileSizeZ);
    LOGF(eINFO, "  Total grid size: %.2f x %.2f", totalGridSizeX, totalGridSizeZ);
    LOGF(eINFO, "  Grid centered at origin, spans X:[%.2f, %.2f] Z:[%.2f, %.2f]",
        gridStartX, gridStartX + totalGridSizeX, gridStartZ,
        gridStartZ + totalGridSizeZ);

    // Generate each floor tile
    for (uint32_t z = 0; z < gridDimZ; ++z)
    {
        for (uint32_t x = 0; x < gridDimX; ++x)
        {
            // Calculate this tile's index in the object array
            uint32_t tileIndex = OBJ_FLOOR_START + (z * gridDimX + x);

            // Calculate this tile's center position in world space
            // Each tile is centered at its position in the grid
            float tileCenterX = gridStartX + (x * tileSizeX) + (tileSizeX * 0.5f);
            float tileCenterZ = gridStartZ + (z * tileSizeZ) + (tileSizeZ * 0.5f);
            vec3  tileCenter = vec3(tileCenterX, 0.0f, tileCenterZ);

            // Generate the mesh for this tile
            generate_plane_mesh(gSimpleMesh[tileIndex],
                                tileSizeX, tileSizeZ,
                                segmentsPerTile, segmentsPerTile, atlasSlot,
                                PLANE_XZ,
                                tileCenter,
                                1.0f); // UV tiling = 1.0

            // Upload to GPU
            upload_mesh_to_gpu(gSimpleMesh[tileIndex],
                gMeshVertexLayout, &pMeshVertexBuffer[tileIndex], &pMeshIndexBuffer[tileIndex],
                gMeshIndexCount[tileIndex], &gMeshCpuMappedVertexData[tileIndex],
                gMeshCpuMappedVertexSize[tileIndex]);

            // Copy animation data (floor tiles don't animate, but we need the structure)
            copy_animation_data(tileIndex, gSimpleMesh[tileIndex]);

            // Set up transform (identity since mesh is already positioned)
            gObjectWorld[tileIndex] = mat4::identity();
            gObjectColor[tileIndex] = vec4(0.7f, 0.9f, 1.0f, 1.0f); // Light blue-ish

            LOGF(eINFO, "  Tile[%u,%u] idx=%u at (%.2f, 0.0, %.2f)",
                x, z, tileIndex, tileCenterX, tileCenterZ);
        }
    }

    LOGF(eINFO, "Floor grid generation complete: %u tiles created", gActiveFloorTiles);
}

// Creates a simple low-poly 3D arrow
//  - Arrow points along +Z axis by default
//  - Consists of: cone tip (8 sides) + cylindrical shaft (8 sides)
static void generate_arrow_mesh(
    SimpleMesh& mesh,
    float length,
    float shaftRadius,
    float tipLength,
    float tipRadius,
    int atlasSlot)
{
    free_simple_mesh(mesh);

    // Arrow geometry parameters
    const uint32_t sides = 8; // Low-poly: 8 sides for cone and cylinder
    const float    shaftLength = length - tipLength;

    // Vertex count: shaft cylinder + cone tip + end caps
    const uint32_t shaftVerts = (sides + 1) * 2; // Cylinder: 2 rings
    const uint32_t coneVerts = sides + 2;        // Cone: ring + tip + center
    const uint32_t capVerts = sides + 1;         // Bottom cap
    const uint32_t vertexCount = shaftVerts + coneVerts + capVerts;

    // Index count: shaft quads + cone triangles + cap triangles
    const uint32_t shaftIndices = sides * 6; // Cylinder sides
    const uint32_t coneIndices = sides * 3;  // Cone sides
    const uint32_t capIndices = sides * 3;   // Bottom cap
    const uint32_t indexCount = shaftIndices + coneIndices + capIndices;

    mesh.vertexCount = vertexCount;
    mesh.indexCount = indexCount;

    // Allocate
    mesh.verts = (float*)tf_malloc(sizeof(float) * 3 * vertexCount);
    mesh.normals = (float*)tf_malloc(sizeof(float) * 3 * vertexCount);
    mesh.colors = (uint8_t*)tf_malloc(sizeof(uint8_t) * 4 * vertexCount);
    mesh.UVs = (float*)tf_malloc(sizeof(float) * 2 * vertexCount);
    mesh.indices = (uint16_t*)tf_malloc(sizeof(uint16_t) * indexCount);

    memset(mesh.verts, 0, sizeof(float) * 3 * vertexCount);
    memset(mesh.normals, 0, sizeof(float) * 3 * vertexCount);
    memset(mesh.colors, 0, sizeof(uint8_t) * 4 * vertexCount);
    memset(mesh.UVs, 0, sizeof(float) * 2 * vertexCount);
    memset(mesh.indices, 0, sizeof(uint16_t) * indexCount);

    const float M_PI = 3.14159265358979323846f;
    uint32_t    vIdx = 0;
    uint32_t    iIdx = 0;

    // SHAFT (Cylinder)
    uint32_t shaftStartIdx = vIdx;

    for (uint32_t ring = 0; ring < 2; ++ring)
    {
        float z = ring == 0 ? 0.0f : shaftLength;

        for (uint32_t s = 0; s <= sides; ++s)
        {
            float angle = (float)s / (float)sides * 2.0f * M_PI;
            float x = shaftRadius * cosf(angle);
            float y = shaftRadius * sinf(angle);

            mesh.verts[vIdx * 3 + 0] = x;
            mesh.verts[vIdx * 3 + 1] = y;
            mesh.verts[vIdx * 3 + 2] = z;

            // Normal points outward radially
            float nx = cosf(angle);
            float ny = sinf(angle);
            mesh.normals[vIdx * 3 + 0] = nx;
            mesh.normals[vIdx * 3 + 1] = ny;
            mesh.normals[vIdx * 3 + 2] = 0.0f;

            // Color and UV
            mesh.colors[vIdx * 4 + 0] = 255;
            mesh.colors[vIdx * 4 + 1] = 255;
            mesh.colors[vIdx * 4 + 2] = 255;
            mesh.colors[vIdx * 4 + 3] = 255;

            float u = (float)s / (float)sides;
            float v = (float)ring;
            float atlasU, atlasV;
            remap_uv_to_atlas(atlasSlot, u, v, &atlasU, &atlasV);

            mesh.UVs[vIdx * 2 + 0] = atlasU;
            mesh.UVs[vIdx * 2 + 1] = atlasV;

            ++vIdx;
        }
    }

    // Shaft indices
    for (uint32_t s = 0; s < sides; ++s)
    {
        uint32_t i0 = shaftStartIdx + s;
        uint32_t i1 = shaftStartIdx + s + 1;
        uint32_t i2 = shaftStartIdx + (sides + 1) + s;
        uint32_t i3 = shaftStartIdx + (sides + 1) + s + 1;

        mesh.indices[iIdx++] = (uint16_t)i0;
        mesh.indices[iIdx++] = (uint16_t)i2;
        mesh.indices[iIdx++] = (uint16_t)i1;

        mesh.indices[iIdx++] = (uint16_t)i1;
        mesh.indices[iIdx++] = (uint16_t)i2;
        mesh.indices[iIdx++] = (uint16_t)i3;
    }

    // CONE TIP
    uint32_t coneStartIdx = vIdx;
    float    coneBaseZ = shaftLength;
    float    coneTipZ = shaftLength + tipLength;

    // Cone base ring
    for (uint32_t s = 0; s <= sides; ++s)
    {
        float angle = (float)s / (float)sides * 2.0f * M_PI;
        float x = tipRadius * cosf(angle);
        float y = tipRadius * sinf(angle);

        mesh.verts[vIdx * 3 + 0] = x;
        mesh.verts[vIdx * 3 + 1] = y;
        mesh.verts[vIdx * 3 + 2] = coneBaseZ;

        // Approximate cone normal
        float nx = cosf(angle);
        float ny = sinf(angle);
        float nz = tipRadius / tipLength; // Slope contribution
        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        mesh.normals[vIdx * 3 + 0] = nx / len;
        mesh.normals[vIdx * 3 + 1] = ny / len;
        mesh.normals[vIdx * 3 + 2] = nz / len;

        mesh.colors[vIdx * 4 + 0] = 255;
        mesh.colors[vIdx * 4 + 1] = 255;
        mesh.colors[vIdx * 4 + 2] = 255;
        mesh.colors[vIdx * 4 + 3] = 255;

        float u = (float)s / (float)sides;
        float v = 0.0f;
        float atlasU, atlasV;
        remap_uv_to_atlas(atlasSlot, u, v, &atlasU, &atlasV);

        mesh.UVs[vIdx * 2 + 0] = atlasU;
        mesh.UVs[vIdx * 2 + 1] = atlasV;

        ++vIdx;
    }

    // Cone tip vertex
    uint32_t tipIdx = vIdx;
    mesh.verts[vIdx * 3 + 0] = 0.0f;
    mesh.verts[vIdx * 3 + 1] = 0.0f;
    mesh.verts[vIdx * 3 + 2] = coneTipZ;

    mesh.normals[vIdx * 3 + 0] = 0.0f;
    mesh.normals[vIdx * 3 + 1] = 0.0f;
    mesh.normals[vIdx * 3 + 2] = 1.0f;

    mesh.colors[vIdx * 4 + 0] = 255;
    mesh.colors[vIdx * 4 + 1] = 255;
    mesh.colors[vIdx * 4 + 2] = 255;
    mesh.colors[vIdx * 4 + 3] = 255;

    float atlasU, atlasV;
    remap_uv_to_atlas(atlasSlot, 0.5f, 1.0f, &atlasU, &atlasV);
    mesh.UVs[vIdx * 2 + 0] = atlasU;
    mesh.UVs[vIdx * 2 + 1] = atlasV;
    ++vIdx;

    // Cone indices
    for (uint32_t s = 0; s < sides; ++s)
    {
        uint32_t i0 = coneStartIdx + s;
        uint32_t i1 = coneStartIdx + s + 1;

        mesh.indices[iIdx++] = (uint16_t)i0;
        mesh.indices[iIdx++] = (uint16_t)tipIdx;
        mesh.indices[iIdx++] = (uint16_t)i1;
    }

    // BOTTOM CAP (closing the cylinder)
    uint32_t capStartIdx = vIdx;

    // Center vertex
    mesh.verts[vIdx * 3 + 0] = 0.0f;
    mesh.verts[vIdx * 3 + 1] = 0.0f;
    mesh.verts[vIdx * 3 + 2] = 0.0f;

    mesh.normals[vIdx * 3 + 0] = 0.0f;
    mesh.normals[vIdx * 3 + 1] = 0.0f;
    mesh.normals[vIdx * 3 + 2] = -1.0f;

    mesh.colors[vIdx * 4 + 0] = 255;
    mesh.colors[vIdx * 4 + 1] = 255;
    mesh.colors[vIdx * 4 + 2] = 255;
    mesh.colors[vIdx * 4 + 3] = 255;

    remap_uv_to_atlas(atlasSlot, 0.5f, 0.5f, &atlasU, &atlasV);
    mesh.UVs[vIdx * 2 + 0] = atlasU;
    mesh.UVs[vIdx * 2 + 1] = atlasV;
    uint32_t centerIdx = vIdx;
    ++vIdx;

    // Cap ring vertices
    for (uint32_t s = 0; s < sides; ++s)
    {
        float angle = (float)s / (float)sides * 2.0f * M_PI;
        float x = shaftRadius * cosf(angle);
        float y = shaftRadius * sinf(angle);

        mesh.verts[vIdx * 3 + 0] = x;
        mesh.verts[vIdx * 3 + 1] = y;
        mesh.verts[vIdx * 3 + 2] = 0.0f;

        mesh.normals[vIdx * 3 + 0] = 0.0f;
        mesh.normals[vIdx * 3 + 1] = 0.0f;
        mesh.normals[vIdx * 3 + 2] = -1.0f;

        mesh.colors[vIdx * 4 + 0] = 255;
        mesh.colors[vIdx * 4 + 1] = 255;
        mesh.colors[vIdx * 4 + 2] = 255;
        mesh.colors[vIdx * 4 + 3] = 255;

        float u = (cosf(angle) + 1.0f) * 0.5f;
        float v = (sinf(angle) + 1.0f) * 0.5f;
        remap_uv_to_atlas(atlasSlot, u, v, &atlasU, &atlasV);

        mesh.UVs[vIdx * 2 + 0] = atlasU;
        mesh.UVs[vIdx * 2 + 1] = atlasV;
        ++vIdx;
    }

    // Cap indices
    for (uint32_t s = 0; s < sides; ++s)
    {
        uint32_t i0 = capStartIdx + 1 + s;
        uint32_t i1 = capStartIdx + 1 + ((s + 1) % sides);

        mesh.indices[iIdx++] = (uint16_t)centerIdx;
        mesh.indices[iIdx++] = (uint16_t)i1;
        mesh.indices[iIdx++] = (uint16_t)i0;
    }
}

// Creates sphere mesh for collision sphere
static void generate_sphere_mesh(
    SimpleMesh& mesh,
    float radius,
    uint32_t segments,
    uint32_t rings,
    int atlasSlot)
{
    // Param validation
    if (segments < 3)
        segments = 3;
    if (rings < 2)
        rings = 2;

    free_simple_mesh(mesh);

    const uint32_t cols = segments + 1;
    const uint32_t rows = rings + 1;
    const uint32_t vertexCount = rows * cols;
    const uint32_t indexCount = segments * 6 * rings;

    mesh.vertexCount = vertexCount;
    mesh.indexCount = indexCount;

    // Allocate buffers
    mesh.verts = (float*)tf_malloc(sizeof(float) * 3 * vertexCount);
    mesh.normals = (float*)tf_malloc(sizeof(float) * 3 * vertexCount);
    mesh.colors = (uint8_t*)tf_malloc(sizeof(uint8_t) * 4 * vertexCount);
    mesh.UVs = (float*)tf_malloc(sizeof(float) * 2 * vertexCount);
    mesh.indices = (uint16_t*)tf_malloc(sizeof(uint16_t) * indexCount);

    memset(mesh.verts, 0, sizeof(float) * 3 * vertexCount);
    memset(mesh.normals, 0, sizeof(float) * 3 * vertexCount);
    memset(mesh.colors, 0, sizeof(uint8_t) * 4 * vertexCount);
    memset(mesh.UVs, 0, sizeof(float) * 2 * vertexCount);
    memset(mesh.indices, 0, sizeof(uint16_t) * indexCount);

    // Vertices
    // Remapped via slot 0 (marble) so the full marble tile covers the whole sphere.
    const float M_PI = 3.14159265358979323846f;

    // Which atlas slot this mesh samples — change this to swap textures.
    const int SPHERE_ATLAS_SLOT = atlasSlot;

    for (uint32_t row = 0; row < rows; ++row)
    {
        float v = (float)row / (float)rings;
        float theta = v * (float)M_PI;

        float sinTheta = sinf(theta);
        float cosTheta = cosf(theta);

        for (uint32_t col = 0; col < cols; ++col)
        {
            float u = (float)col / (float)segments;
            float phi = u * 2.0f * (float)M_PI;

            float sinPhi = sinf(phi);
            float cosPhi = cosf(phi);

            // Position
            float px = radius * sinTheta * sinPhi;
            float py = radius * cosTheta;
            float pz = radius * sinTheta * cosPhi;

            uint32_t idx = row * cols + col;

            mesh.verts[idx * 3 + 0] = px;
            mesh.verts[idx * 3 + 1] = py;
            mesh.verts[idx * 3 + 2] = pz;

            // Normal
            mesh.normals[idx * 3 + 0] = px / radius;
            mesh.normals[idx * 3 + 1] = py / radius;
            mesh.normals[idx * 3 + 2] = pz / radius;

            // Colour – white
            mesh.colors[idx * 4 + 0] = 255;
            mesh.colors[idx * 4 + 1] = 255;
            mesh.colors[idx * 4 + 2] = 255;
            mesh.colors[idx * 4 + 3] = 255;

            // UVs – remap local (u,v) into the atlas tile for MARBLE
            float atlasU, atlasV;
            remap_uv_to_atlas(SPHERE_ATLAS_SLOT, u, v, &atlasU, &atlasV);
            mesh.UVs[idx * 2 + 0] = atlasU;
            mesh.UVs[idx * 2 + 1] = atlasV;
        }
    }

    // Indices
    uint32_t idxOut = 0;

    // Top cap
    for (uint32_t s = 0; s < segments; ++s)
    {
        uint16_t pole = (uint16_t)(0 * cols + s);
        uint16_t curr = (uint16_t)(1 * cols + s);
        uint16_t next = (uint16_t)(1 * cols + (s + 1));

        mesh.indices[idxOut++] = pole;
        mesh.indices[idxOut++] = next;
        mesh.indices[idxOut++] = curr;
    }

    // Middle bands
    for (uint32_t r = 1; r < rings - 1; ++r)
    {
        for (uint32_t s = 0; s < segments; ++s)
        {
            uint16_t i0 = (uint16_t)(r * cols + s);
            uint16_t i1 = (uint16_t)(r * cols + (s + 1));
            uint16_t i2 = (uint16_t)((r + 1) * cols + s);
            uint16_t i3 = (uint16_t)((r + 1) * cols + (s + 1));

            mesh.indices[idxOut++] = i0;
            mesh.indices[idxOut++] = i2;
            mesh.indices[idxOut++] = i1;

            mesh.indices[idxOut++] = i1;
            mesh.indices[idxOut++] = i2;
            mesh.indices[idxOut++] = i3;
        }
    }

    // Bottom cap
    for (uint32_t s = 0; s < segments; ++s)
    {
        uint16_t curr = (uint16_t)((rings - 1) * cols + s);
        uint16_t next = (uint16_t)((rings - 1) * cols + (s + 1));
        uint16_t pole = (uint16_t)(rings * cols + s);

        mesh.indices[idxOut++] = curr;
        mesh.indices[idxOut++] = next;
        mesh.indices[idxOut++] = pole;
    }
}

// Create plane mesh for floor and cloth
static void generate_plane_mesh(
    SimpleMesh& mesh,
    float width,
    float depth,
    uint32_t xSegments,
    uint32_t zSegments,
    int atlasSlot,
    PlaneOrientation orientation,
    vec3 worldCenter,
    float uvTiling)
{
    // Param validation
    if (xSegments < 1)
        xSegments = 1;
    if (zSegments < 1)
        zSegments = 1;

    free_simple_mesh(mesh);

    const uint32_t vertX = xSegments + 1;
    const uint32_t vertZ = zSegments + 1;

    const uint32_t vertexCount = vertX * vertZ;
    const uint32_t indexCount = xSegments * zSegments * 6;

    mesh.vertexCount = vertexCount;
    mesh.indexCount = indexCount;

    // Allocate buffers
    mesh.verts = (float*)tf_malloc(sizeof(float) * 3 * vertexCount);
    mesh.normals = (float*)tf_malloc(sizeof(float) * 3 * vertexCount);
    mesh.colors = (uint8_t*)tf_malloc(sizeof(uint8_t) * 4 * vertexCount);
    mesh.UVs = (float*)tf_malloc(sizeof(float) * 2 * vertexCount);
    mesh.indices = (uint16_t*)tf_malloc(sizeof(uint16_t) * indexCount);

    // Allocate edge arrays
    mesh.edgeIndices[0] = (uint16_t*)tf_malloc(sizeof(uint16_t) * vertX); // top
    mesh.edgeIndices[1] = (uint16_t*)tf_malloc(sizeof(uint16_t) * vertX); // bottom
    mesh.edgeIndices[2] = (uint16_t*)tf_malloc(sizeof(uint16_t) * vertZ); // left
    mesh.edgeIndices[3] = (uint16_t*)tf_malloc(sizeof(uint16_t) * vertZ); // right

    mesh.edgeCounts[0] = vertX;
    mesh.edgeCounts[1] = vertX;
    mesh.edgeCounts[2] = vertZ;
    mesh.edgeCounts[3] = vertZ;

    memset(mesh.verts, 0, sizeof(float) * 3 * vertexCount);
    memset(mesh.normals, 0, sizeof(float) * 3 * vertexCount);
    memset(mesh.colors, 0, sizeof(uint8_t) * 4 * vertexCount);
    memset(mesh.UVs, 0, sizeof(float) * 2 * vertexCount);
    memset(mesh.indices, 0, sizeof(uint16_t) * indexCount);

    const float halfW = width * 0.5f;
    const float halfD = depth * 0.5f;

    const int PLANE_ATLAS_SLOT = atlasSlot;

    // Vertices - Generate based on orientation
    uint32_t vtx = 0;
    for (uint32_t z = 0; z < vertZ; ++z)
    {
        float fz = float(z) / float(zSegments);
        float localZ = -halfD + fz * depth;

        for (uint32_t x = 0; x < vertX; ++x)
        {
            float fx = float(x) / float(xSegments);
            float localX = -halfW + fx * width;

            float px, py, pz;
            float nx, ny, nz;

            switch (orientation)
            {
            case PLANE_XZ:
                px = worldCenter.x + localX;
                py = worldCenter.y;
                pz = worldCenter.z + localZ;
                nx = 0.0f;
                ny = 1.0f;
                nz = 0.0f;
                break;

            case PLANE_XY:
                px = worldCenter.x + localX;
                py = worldCenter.y + localZ;
                pz = worldCenter.z;
                nx = 0.0f;
                ny = 0.0f;
                nz = 1.0f;
                break;

            case PLANE_YZ:
                px = worldCenter.x;
                py = worldCenter.y + localX;
                pz = worldCenter.z + localZ;
                nx = 1.0f;
                ny = 0.0f;
                nz = 0.0f;
                break;

            default:
                px = worldCenter.x + localX;
                py = worldCenter.y;
                pz = worldCenter.z + localZ;
                nx = 0.0f;
                ny = 1.0f;
                nz = 0.0f;
                break;
            }

            mesh.verts[vtx * 3 + 0] = px;
            mesh.verts[vtx * 3 + 1] = py;
            mesh.verts[vtx * 3 + 2] = pz;

            mesh.normals[vtx * 3 + 0] = nx;
            mesh.normals[vtx * 3 + 1] = ny;
            mesh.normals[vtx * 3 + 2] = nz;

            mesh.colors[vtx * 4 + 0] = 255;
            mesh.colors[vtx * 4 + 1] = 255;
            mesh.colors[vtx * 4 + 2] = 255;
            mesh.colors[vtx * 4 + 3] = 255;

            float tiledU = fmodf(fx * uvTiling, 1.0f);
            float tiledV = fmodf(fz * uvTiling, 1.0f);

            float atlasU, atlasV;
            remap_uv_to_atlas(PLANE_ATLAS_SLOT, tiledU, tiledV, &atlasU, &atlasV);

            mesh.UVs[vtx * 2 + 0] = atlasU;
            mesh.UVs[vtx * 2 + 1] = atlasV;

            ++vtx;
        }
    }

    // Indices
    uint32_t idx = 0;
    for (uint32_t z = 0; z < zSegments; ++z)
    {
        for (uint32_t x = 0; x < xSegments; ++x)
        {
            uint16_t i0 = uint16_t(z * vertX + x);
            uint16_t i1 = uint16_t(i0 + 1);
            uint16_t i2 = uint16_t(i0 + vertX);
            uint16_t i3 = uint16_t(i2 + 1);

            mesh.indices[idx++] = i0;
            mesh.indices[idx++] = i2;
            mesh.indices[idx++] = i1;

            mesh.indices[idx++] = i1;
            mesh.indices[idx++] = i2;
            mesh.indices[idx++] = i3;
        }
    }

    // ---------------------------------------------------------------------
    // Populate Edge Indices
    // NOTE: For PLANE_XY (hanging cloth):
    //   - edgeIndices[0] = BOTTOM edge (z=0 maps to lower Y)
    //   - edgeIndices[1] = TOP edge (z=zSegments maps to higher Y)
    //   - edgeIndices[2] = LEFT edge (x=0)
    //   - edgeIndices[3] = RIGHT edge (x=xSegments)
    // Edge 0: z=0 row (BOTTOM for PLANE_XY, TOP for PLANE_XZ)
    for (uint32_t x = 0; x < vertX; ++x)
    {
        mesh.edgeIndices[0][x] = (uint16_t)(0 * vertX + x);
    }

    // Edge 1: z=zSegments row (TOP for PLANE_XY, BOTTOM for PLANE_XZ)
    for (uint32_t x = 0; x < vertX; ++x)
    {
        mesh.edgeIndices[1][x] = (uint16_t)(zSegments * vertX + x);
    }

    // Edge 2: x=0 column (LEFT edge)
    for (uint32_t z = 0; z < vertZ; ++z)
    {
        mesh.edgeIndices[2][z] = (uint16_t)(z * vertX + 0);
    }

    // Edge 3: x=xSegments column (RIGHT edge)
    for (uint32_t z = 0; z < vertZ; ++z)
    {
        mesh.edgeIndices[3][z] = (uint16_t)(z * vertX + xSegments);
    }

    // Find Center Vertex
    uint32_t centerX = xSegments / 2;
    uint32_t centerZ = zSegments / 2;
    mesh.centerIndex = (uint16_t)(centerZ * vertX + centerX);
}

// Copy edge/center indices from SimpleMesh to global animation data
static void copy_animation_data(
    uint32_t objIndex,
    const SimpleMesh& mesh)
{
    MeshAnimationData& animData = gAnimData[objIndex];

    // Free any existing data
    for (int i = 0; i < 4; ++i)
    {
        if (animData.edgeIndices[i])
        {
            tf_free(animData.edgeIndices[i]);
            animData.edgeIndices[i] = nullptr;
        }
    }

    // Check if mesh has edge data
    animData.hasEdgeData = (mesh.edgeIndices[0] != nullptr);

    if (!animData.hasEdgeData)
        return;

    // Copy edge indices
    for (int i = 0; i < 4; ++i)
    {
        animData.edgeCounts[i] = mesh.edgeCounts[i];
        if (mesh.edgeIndices[i] && mesh.edgeCounts[i] > 0)
        {
            animData.edgeIndices[i] = (uint16_t*)tf_malloc(sizeof(uint16_t) * mesh.edgeCounts[i]);
            memcpy(animData.edgeIndices[i], mesh.edgeIndices[i], sizeof(uint16_t) * mesh.edgeCounts[i]);
        }
    }

    // Copy center index
    animData.centerIndex = mesh.centerIndex;
}
