#pragma once

#include "Common.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
// CLOTH SIMULATION MODULE - Position-Based Dynamics cloth simulator using Verlet integration
// - Simulates hanging cloth as a grid of particles connected by distance constraints
// - Uses constraint relaxation to maintain structural integrity and handle collisions
//
// BASED ON: "Advanced Character Physics" by Thomas Jakobsen (GDC 2001)
//          Paper developed for IO Interactive's Hitman: Codename 47
//          Video tutorial: https://www.youtube.com/watch?v=erLT9HsllJU&t=278s
//
// MAIN ALGORITHM: Verlet Integration + Iterative Constraint Solver
// • AccumulateForces() - Apply gravity to all particles
// • Verlet() - Update particle positions using Verlet integration (x_new = x + damping*(x-x_old) + a*dt²)
// • SatisfyConstraints() - Iteratively enforce distance constraints, edge pins, ground/sphere collisions
// • RecalculateNormals() - Compute vertex normals from face normals for rendering
//
// CORE SETUP:
// • CreateFromPlaneMesh() - Generates horizontal, vertical, and diagonal stick constraints from mesh grid
// • Edge pinning constraints keep top/bottom edges fixed at specific world positions
//
// COLLISION SUPPORT:
// • Ground plane collision (prevents cloth from falling through Y=0)
// • Sphere collision (pushes particles outside sphere radius)
//////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////
//                                      Enums
//////////////////////////////////////////////////////////////////////////////////////////////////


// Edge indices for hanging cloth (PLANE_XY orientation)
// Note: For PLANE_XY, the 'z' parameter in mesh generation maps to Y (height)
enum ClothEdge
{
    CLOTH_EDGE_BOTTOM = 0, // z=0 → Lower Y position (bottom of hanging cloth)
    CLOTH_EDGE_TOP = 1,    // z=zSegments → Higher Y position (top of hanging cloth)
    CLOTH_EDGE_RIGHT = 2,  // x=0 → Left side
    CLOTH_EDGE_LEFT = 3    // x=xSegments → Right side
};

//////////////////////////////////////////////////////////////////////////////////////////////////
//                                      Constants/Defines
//////////////////////////////////////////////////////////////////////////////////////////////////


// Maximum cloth dimensions
#define MAX_CLOTH_SEGMENTS    512
#define MAX_CLOTH_VERTICES    (MAX_CLOTH_SEGMENTS * MAX_CLOTH_SEGMENTS)
#define MAX_EDGE_POINTS       (MAX_CLOTH_SEGMENTS + 1) // Max points per edge
// Constraints: Horizontal + Vertical + Diagonal (2 per quad)
// = (vertZ * xSegments) + (vertX * zSegments) + (2 * xSegments * zSegments)
// = roughly 4 * segments^2 for square cloth
#define MAX_CLOTH_CONSTRAINTS ((MAX_CLOTH_SEGMENTS * MAX_CLOTH_SEGMENTS * 4) + (MAX_CLOTH_SEGMENTS * 2))


//////////////////////////////////////////////////////////////////////////////////////////////////
//                                      Structs
//////////////////////////////////////////////////////////////////////////////////////////////////

// StickConstraint - represents a distance constraint between two particles
struct StickConstraint
{
    uint32_t particleA;  // Index of first particle
    uint32_t particleB;  // Index of second particle
    float    restLength; // Rest distance between particles
};

// EdgeConstraint - pins an edge vertex to a specific world position
struct EdgeConstraint
{
    vec3     pos;   // World position where this vertex is pinned
    uint32_t index; // Index of the vertex in the mesh
};

// SphereCollider - sphere collision object for cloth
struct SphereCollider
{
    vec3  center; // World space center of the sphere
    float radius; // World space radius (after scale applied)
};

// Cloth - manages cloth simulation with stick constraints
class Cloth
{
public:
    Cloth()
    {
        m_constraintCount = 0;
        m_topEdgeCount = 0;
        m_bottomEdgeCount = 0;
        m_particleCount = 0;

        // Default gravity (small value pointing down)
        m_vGravity.x = 0.0f;
        m_vGravity.y = -10.0f;
        m_vGravity.z = 0.0f;

        m_numIterations = 15;  // Number of constraint solver iterations
        m_groundHeight = 0.0f; // Ground at Y=0

        m_hasSphereCollider = false;
        m_releasePinContraints = false;
    }

    void ReleasePinCloth()
    {
        m_releasePinContraints = !m_releasePinContraints;
    }

    // Create cloth constraints from a plane mesh
    void CreateFromPlaneMesh(const SimpleMesh& mesh, uint32_t xSegments, uint32_t zSegments)
    {
        m_constraintCount = 0;
        m_topEdgeCount = 0;
        m_bottomEdgeCount = 0;

        // Store dimensions for normal calculation
        m_xSegments = xSegments;
        m_zSegments = zSegments;

        // Safety check
        if (xSegments > MAX_CLOTH_SEGMENTS || zSegments > MAX_CLOTH_SEGMENTS)
        {
            // Log error or assert
            return;
        }

        const uint32_t vertX = xSegments + 1;
        const uint32_t vertZ = zSegments + 1;
        m_particleCount = vertX * vertZ;

        // Copy initial positions from mesh to current and previous positions
        for (uint32_t i = 0; i < m_particleCount; ++i)
        {
            float* meshPos = &mesh.verts[i * 3];

            // Current position
            m_x[i].x = meshPos[0];
            m_x[i].y = meshPos[1];
            m_x[i].z = meshPos[2];

            // Previous position (same as current initially)
            m_oldx[i].x = meshPos[0];
            m_oldx[i].y = meshPos[1];
            m_oldx[i].z = meshPos[2];

            // Zero out force accumulator
            m_a[i].x = 0.0f;
            m_a[i].y = 0.0f;
            m_a[i].z = 0.0f;

            // Initialize normal (pointing forward +Z for PLANE_XY)
            m_normals[i].x = 0.0f; // ← ADD
            m_normals[i].y = 0.0f; // ← ADD
            m_normals[i].z = 1.0f; // ← ADD (pointing forward)
        }

        ///////////////////////////
        // Structural constraints
        ///////////////////////////

        // Generate horizontal constraints (e0, e1, e2, etc.)
        // Each row has xSegments edges connecting adjacent vertices
        for (uint32_t z = 0; z < vertZ; ++z)
        {
            for (uint32_t x = 0; x < xSegments; ++x)
            {
                uint32_t pA = z * vertX + x;
                uint32_t pB = z * vertX + (x + 1);

                float restLength = CalculateDistance(mesh, pA, pB);

                StickConstraint& constraint = m_stickConstraints[m_constraintCount++];
                constraint.particleA = pA;
                constraint.particleB = pB;
                constraint.restLength = restLength;
            }
        }

        // Generate vertical constraints (f0, f1, f2, etc.)
        // Each column has zSegments edges connecting adjacent vertices
        for (uint32_t x = 0; x < vertX; ++x)
        {
            for (uint32_t z = 0; z < zSegments; ++z)
            {
                uint32_t pA = z * vertX + x;
                uint32_t pB = (z + 1) * vertX + x;

                float restLength = CalculateDistance(mesh, pA, pB);

                StickConstraint& constraint = m_stickConstraints[m_constraintCount++];
                constraint.particleA = pA;
                constraint.particleB = pB;
                constraint.restLength = restLength;
            }
        }

        // Diagonals-> These prevent the cloth from collapsing/shearing
        // Optional
        if (true)
        {
            for (uint32_t z = 0; z < zSegments; ++z)
            {
                for (uint32_t x = 0; x < xSegments; ++x)
                {
                    // Four corners of current quad:
                    //   p0 --- p1
                    //   |   X  |
                    //   p2 --- p3

                    uint32_t p0 = z * vertX + x;
                    uint32_t p1 = z * vertX + (x + 1);
                    uint32_t p2 = (z + 1) * vertX + x;
                    uint32_t p3 = (z + 1) * vertX + (x + 1);

                    // Diagonal 1: p0 to p3 (top-left to bottom-right)
                    {
                        float            restLength = CalculateDistance(mesh, p0, p3);
                        StickConstraint& constraint = m_stickConstraints[m_constraintCount++];
                        constraint.particleA = p0;
                        constraint.particleB = p3;
                        constraint.restLength = restLength;
                    }

                    // Diagonal 2: p1 to p2 (top-right to bottom-left)
                    {
                        float            restLength = CalculateDistance(mesh, p1, p2);
                        StickConstraint& constraint = m_stickConstraints[m_constraintCount++];
                        constraint.particleA = p1;
                        constraint.particleB = p2;
                        constraint.restLength = restLength;
                    }
                }
            }
        }

        // Edge constraints-> Pin top and bottom edges to their initial world positions

        // Top edge (z = zSegments, higher Y for PLANE_XY)
        for (uint32_t x = 0; x < vertX; ++x)
        {
            uint32_t vertIdx = zSegments * vertX + x;

            EdgeConstraint& edge = m_topEdgeConstraints[m_topEdgeCount];
            edge.index = vertIdx;

            // Get world position from mesh
            float* pos = &mesh.verts[vertIdx * 3];
            edge.pos.x = pos[0];
            edge.pos.y = pos[1];
            edge.pos.z = pos[2];

            // Store the index for quick lookup
            m_topEdgeIndexes[m_topEdgeCount] = vertIdx;

            m_topEdgeCount++;
        }

        // Bottom edge (z = 0, lower Y for PLANE_XY)
        for (uint32_t x = 0; x < vertX; ++x)
        {
            uint32_t vertIdx = 0 * vertX + x;

            EdgeConstraint& edge = m_bottomEdgeConstraints[m_bottomEdgeCount];
            edge.index = vertIdx;

            // Get world position from mesh
            float* pos = &mesh.verts[vertIdx * 3];
            edge.pos.x = pos[0];
            edge.pos.y = pos[1];
            edge.pos.z = pos[2];

            // Store the index for quick lookup
            m_bottomEdgeIndexes[m_bottomEdgeCount] = vertIdx;

            m_bottomEdgeCount++;
        }
    }

    // Get all stick constraints
    const StickConstraint* GetConstraints() const { return m_stickConstraints; }
    uint32_t               GetConstraintCount() const { return m_constraintCount; }

    // Get edge constraints
    const EdgeConstraint* GetTopEdgeConstraints() const { return m_topEdgeConstraints; }
    uint32_t              GetTopEdgeCount() const { return m_topEdgeCount; }

    const EdgeConstraint* GetBottomEdgeConstraints() const { return m_bottomEdgeConstraints; }
    uint32_t              GetBottomEdgeCount() const { return m_bottomEdgeCount; }

    // Get particle data
    vec3*    GetCurrentPositions() { return m_x; }
    vec3*    GetPreviousPositions() { return m_oldx; }
    vec3*    GetForceAccumulators() { return m_a; }
    vec3*    GetVertexNormals() { return m_normals; }
    uint32_t GetParticleCount() const { return m_particleCount; }

    // Get/Set gravity
    const vec3& GetGravity() const { return m_vGravity; }
    void        SetGravity(const vec3& gravity) { m_vGravity = gravity; }

    // Pass the center and already-computed world radius
    void SetSphereCollider(const vec3& center, float radius)
    {
        m_sphereCollider.center = center;
        m_sphereCollider.radius = radius;
        m_hasSphereCollider = true;
    }

    // Recalculate vertex normals by averaging face normals
    void RecalculateNormals()
    {
        const uint32_t vertX = m_xSegments + 1;

        // Zero out all normals
        for (uint32_t i = 0; i < m_particleCount; i++)
        {
            m_normals[i].x = 0.0f;
            m_normals[i].y = 0.0f;
            m_normals[i].z = 0.0f;
        }

        // Loop through all quads
        for (uint32_t z = 0; z < m_zSegments; z++)
        {
            for (uint32_t x = 0; x < m_xSegments; x++)
            {
                // Four corners of this quad:
                //   v0 --- v1
                //   |  \   |
                //   v2 --- v3

                uint32_t v0 = z * vertX + x;
                uint32_t v1 = z * vertX + (x + 1);
                uint32_t v2 = (z + 1) * vertX + x;
                uint32_t v3 = (z + 1) * vertX + (x + 1);

                // Triangle 1: v0, v2, v1 (counter-clockwise from front)
                vec3 normal1 = CalculateFaceNormal(m_x[v0], m_x[v2], m_x[v1]);

                // Triangle 2: v1, v2, v3 (counter-clockwise from front)
                vec3 normal2 = CalculateFaceNormal(m_x[v1], m_x[v2], m_x[v3]);

                // Accumulate to all 4 vertices
                m_normals[v0].x += normal1.x;
                m_normals[v0].y += normal1.y;
                m_normals[v0].z += normal1.z;
                m_normals[v1].x += normal1.x + normal2.x;
                m_normals[v1].y += normal1.y + normal2.y;
                m_normals[v1].z += normal1.z + normal2.z;
                m_normals[v2].x += normal1.x + normal2.x;
                m_normals[v2].y += normal1.y + normal2.y;
                m_normals[v2].z += normal1.z + normal2.z;
                m_normals[v3].x += normal2.x;
                m_normals[v3].y += normal2.y;
                m_normals[v3].z += normal2.z;
            }
        }

        // Normalize all vertex normals
        for (uint32_t i = 0; i < m_particleCount; i++)
        {
            float length = sqrtf(
                m_normals[i].x * m_normals[i].x +
                m_normals[i].y * m_normals[i].y +
                m_normals[i].z * m_normals[i].z);

            if (length > 0.000001f)
            {
                m_normals[i].x /= length;
                m_normals[i].y /= length;
                m_normals[i].z /= length;

                // Flip normals
                m_normals[i].x = -m_normals[i].x;
                m_normals[i].y = -m_normals[i].y;
                m_normals[i].z = -m_normals[i].z;
            }
        }
    }

    //////////////////////
    // Physics Simulation
    //////////////////////
    void Update(float deltaTime)
    {
        // the more times we run more realsitic but more
        // calculations
        const uint32_t simulationSpeed = 4;
        for (uint32_t ii = 0; ii < simulationSpeed; ii++)
        {
            AccumulateForces();
            Verlet(deltaTime);
            SatisfyConstraints();
        }

        // Recalculate normals after simulation
        RecalculateNormals();
    }

    void AccumulateForces()
    {
        // All particles are influenced by gravity
        for (uint32_t i = 0; i < m_particleCount; i++)
        {
            m_a[i] = m_vGravity;
        }
    }

    void Verlet(float deltaTime)
    {
        float damping = 0.99f; // Damping factor to prevent instability

        for (uint32_t i = 0; i < m_particleCount; i++)
        {
            vec3 temp = m_x[i]; // Store current position

            // Verlet integration: x_new = x + damping * (x - x_old) + a * dt^2
            m_x[i].x = m_x[i].x + damping * (m_x[i].x - m_oldx[i].x) + m_a[i].x * deltaTime * deltaTime;
            m_x[i].y = m_x[i].y + damping * (m_x[i].y - m_oldx[i].y) + m_a[i].y * deltaTime * deltaTime;
            m_x[i].z = m_x[i].z + damping * (m_x[i].z - m_oldx[i].z) + m_a[i].z * deltaTime * deltaTime;

            // Update old position
            m_oldx[i] = temp;
        }
    }

    void SatisfyConstraints()
    {
        // Ground collision constraint (cloth can't go below ground)
        for (uint32_t i = 0; i < m_particleCount; i++)
        {
            if (m_x[i].y < m_groundHeight)
            {
                m_x[i].y = m_groundHeight;
            }
        }

        if (!m_releasePinContraints)
        {
           
            // HACK: bottom edge can never rise above half the cloth height
            // Cloth top is ~40.25, bottom starts at ~0.25, so half = ~20.25
            static const float BOTTOM_EDGE_MAX_Y = 20.25f;
            for (uint32_t i = 0; i < m_bottomEdgeCount; i++)
            {
                uint32_t idx = m_bottomEdgeConstraints[i].index;
                if (m_x[idx].y > BOTTOM_EDGE_MAX_Y)
                {
                    m_x[idx].y = BOTTOM_EDGE_MAX_Y;
                }
            }

            // Pin top edge vertices to their original positions
            for (uint32_t i = 0; i < m_topEdgeCount; i++)
            {
                const EdgeConstraint& edge = m_topEdgeConstraints[i];
                m_x[edge.index] = edge.pos;
            }
        }

        //////////////////////////
        // Relaxation loop
        //////////////////////////
        for (uint32_t iteration = 0; iteration < m_numIterations; iteration++)
        {

            // c1 = - satisfy stick constraints
            for (uint32_t c = 0; c < m_constraintCount; c++)
            {
                const StickConstraint& constraint = m_stickConstraints[c];

                vec3& x1 = m_x[constraint.particleA];
                vec3& x2 = m_x[constraint.particleB];

                // Calculate delta vector
                vec3 delta;
                delta.x = x2.x - x1.x;
                delta.y = x2.y - x1.y;
                delta.z = x2.z - x1.z;

                // Calculate current length
                float deltaLength = sqrtf(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);

                // Avoid division by zero
                if (deltaLength < 0.000001f)
                    continue;

                // Calculate difference from rest length
                float diff = (deltaLength - constraint.restLength) / deltaLength;

                // Move both particles half the correction distance
                x1.x += delta.x * 0.5f * diff;
                x1.y += delta.y * 0.5f * diff;
                x1.z += delta.z * 0.5f * diff;

                x2.x -= delta.x * 0.5f * diff;
                x2.y -= delta.y * 0.5f * diff;
                x2.z -= delta.z * 0.5f * diff;
            }

            // ----------------------------------------------------------------
            // c2 = Sphere collision constraint
            // ----------------------------------------------------------------
            if (m_hasSphereCollider)
            {
                float bias = 0.01f;
                float sphereRadius = m_sphereCollider.radius + bias;

                for (uint32_t i = 0; i < m_particleCount; i++)
                {
                    // Delta from sphere center to particle
                    vec3 delta;
                    delta.x = m_x[i].x - m_sphereCollider.center.x;
                    delta.y = m_x[i].y - m_sphereCollider.center.y;
                    delta.z = m_x[i].z - m_sphereCollider.center.z;

                    // Distance from sphere center to particle
                    float deltaLength = sqrtf(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);

                    // If particle is inside the sphere, project it out to the surface
                    if (deltaLength < sphereRadius)
                    {
                        // Normalize delta to get direction
                        float invLength = 1.0f / deltaLength;
                        vec3  dir;
                        dir.x = delta.x * invLength;
                        dir.y = delta.y * invLength;
                        dir.z = delta.z * invLength;

                        // Project particle to sphere surface
                        m_x[i].x = m_sphereCollider.center.x + dir.x * sphereRadius;
                        m_x[i].y = m_sphereCollider.center.y + dir.y * sphereRadius;
                        m_x[i].z = m_sphereCollider.center.z + dir.z * sphereRadius;
                    }
                }
            }
        }
    }
    // Clear all constraints
    void Clear()
    {
        m_constraintCount = 0;
        m_topEdgeCount = 0;
        m_bottomEdgeCount = 0;
        m_particleCount = 0;
    }

    // Nudge an edge in Z direction by the given amount
    void NudgeEdge(ClothEdge whichEdge, float zNudge)
    {
        EdgeConstraint* constraints = nullptr;
        uint32_t        count = 0;

        // Select which edge
        if (whichEdge == CLOTH_EDGE_TOP)
        {
            constraints = m_topEdgeConstraints; // Get the constraints
            count = m_topEdgeCount;
        }
        else if (whichEdge == CLOTH_EDGE_BOTTOM)
        {
            constraints = m_bottomEdgeConstraints;
            count = m_bottomEdgeCount;
        }
        else
        {
            return;
        }

        // Apply Z nudge to the CONSTRAINT positions (not m_x)
        for (uint32_t i = 0; i < count; i++)
        {
            constraints[i].pos.z += zNudge; // Modify the pin position
        }
    }

private:
    // Stick constraints
    StickConstraint m_stickConstraints[MAX_CLOTH_CONSTRAINTS];
    uint32_t        m_constraintCount;

    // Edge constraints
    EdgeConstraint m_topEdgeConstraints[MAX_EDGE_POINTS];
    uint32_t       m_topEdgeCount;

    EdgeConstraint m_bottomEdgeConstraints[MAX_EDGE_POINTS];
    uint32_t       m_bottomEdgeCount;

    // Edge vertex indices for quick lookup
    uint32_t m_topEdgeIndexes[MAX_EDGE_POINTS];
    uint32_t m_bottomEdgeIndexes[MAX_EDGE_POINTS];

    // Particle simulation data

    vec3     m_x[MAX_CLOTH_VERTICES];    // Current positions
    vec3     m_oldx[MAX_CLOTH_VERTICES]; // Previous positions
    vec3     m_a[MAX_CLOTH_VERTICES];    // Force accumulators
    vec3     m_normals[MAX_CLOTH_VERTICES];
    uint32_t m_particleCount;

    // Mesh grid dimensions (needed for normal calculation)
    uint32_t    m_xSegments;
    uint32_t    m_zSegments;

    // Physics parameters
    vec3 m_vGravity; // Gravity acceleration

    uint32_t m_numIterations; // Number of constraint relaxation iterations
    float    m_groundHeight;  // Ground plane Y position

    SphereCollider  m_sphereCollider;    // Sphere collision object
    bool            m_hasSphereCollider; // Whether we have a sphere to collide with

    bool            m_releasePinContraints;

    // Calculate 3D distance between two vertices in the mesh
    float CalculateDistance(const SimpleMesh& mesh, uint32_t idxA, uint32_t idxB) const
    {
        // Vertices are stored as [x, y, z, x, y, z, ...]
        float* posA = &mesh.verts[idxA * 3];
        float* posB = &mesh.verts[idxB * 3];

        float dx = posB[0] - posA[0];
        float dy = posB[1] - posA[1];
        float dz = posB[2] - posA[2];

        return sqrtf(dx * dx + dy * dy + dz * dz);
    }

private:
    // Calculate face normal from 3 vertices (counter-clockwise winding)
    vec3 CalculateFaceNormal(const vec3& p0, const vec3& p1, const vec3& p2) const
    {
        // Two edges of the triangle
        vec3 edge1, edge2;
        edge1.x = p1.x - p0.x;
        edge1.y = p1.y - p0.y;
        edge1.z = p1.z - p0.z;

        edge2.x = p2.x - p0.x;
        edge2.y = p2.y - p0.y;
        edge2.z = p2.z - p0.z;

        // Cross product: edge1 × edge2
        vec3 normal;
        normal.x = edge1.y * edge2.z - edge1.z * edge2.y;
        normal.y = edge1.z * edge2.x - edge1.x * edge2.z;
        normal.z = edge1.x * edge2.y - edge1.y * edge2.x;

        // Normalize
        float length = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        if (length > 0.000001f)
        {
            normal.x /= length;
            normal.y /= length;
            normal.z /= length;
        }

        return normal;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////
//                                      Globals
//////////////////////////////////////////////////////////////////////////////////////////////////

static Cloth gCloth = {};