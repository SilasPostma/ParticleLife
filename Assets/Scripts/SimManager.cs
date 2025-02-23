using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

struct Particle
{
    public Vector2 position;
    public Vector2 velocity;
    public Color color;
    public int colorIndex;
}

public class SimManager : MonoBehaviour
{
    // variables
    [SerializeField] private int numParticles;
    [SerializeField] private float forceFactor;
    [SerializeField] private float rMax;
    [SerializeField] private float beta;
    [SerializeField] private float maxSpeed;
    [SerializeField] private List<Color> particleColors;

    private float timeStep, frictionFactor, frictionHalve;
    private int numColors;

    private Particle[] particles;
    private float[] frictionMatrix;

    public Mesh particleMesh;
    public Material particleMaterial;

    private float screenLeft, screenRight, screenBottom, screenTop;

    private MaterialPropertyBlock propertyBlock;
    private Matrix4x4[] particleMatrices;
    private BatchData[] batchCache;

    private float width, height, halfWidth, halfHeight;

    public ComputeShader computeShader;
    private ComputeBuffer particleBuffer;
    private ComputeBuffer frictionMatrixBuffer;

    // Grid buffers
    private ComputeBuffer gridStartBuffer;
    private ComputeBuffer gridEndBuffer;
    private ComputeBuffer particleIndexBuffer;
    private ComputeBuffer gridBuffer; // if needed

    private Vector2Int gridSize;
    private float cellSize;

    private void Start()
    {
        propertyBlock = new MaterialPropertyBlock();
        particleMatrices = new Matrix4x4[numParticles];

        Camera mainCamera = Camera.main;
        screenLeft = mainCamera.ScreenToWorldPoint(new Vector3(0, 0, 0)).x + 0.5f;
        screenRight = mainCamera.ScreenToWorldPoint(new Vector3(Screen.width, 0, 0)).x - 0.5f;
        screenBottom = mainCamera.ScreenToWorldPoint(new Vector3(0, 0, 0)).y + 0.5f;
        screenTop = mainCamera.ScreenToWorldPoint(new Vector3(0, Screen.height, 0)).y - 0.5f;

        width = screenRight - screenLeft;
        height = screenTop - screenBottom;
        halfWidth = width * 0.5f;
        halfHeight = height * 0.5f;

        numColors = particleColors.Count;
        timeStep = 0.01f;
        frictionHalve = 0.08f;
        frictionFactor = (float)Math.Pow(0.5f, timeStep / frictionHalve);
        frictionMatrix = initRandomMatrixFlat(numColors);

        particles = new Particle[numParticles];
        SpawnParticles(numParticles, numColors);

        int batchSize = 1023;
        int batches = Mathf.CeilToInt((float)numParticles / batchSize);
        batchCache = new BatchData[batches];
        for (int i = 0; i < batches; i++)
        {
            int size = Mathf.Min(batchSize, numParticles - (i * batchSize));
            batchCache[i] = new BatchData(size);
        }

        particleBuffer = new ComputeBuffer(numParticles, 36);
        frictionMatrixBuffer = new ComputeBuffer(numColors * numColors, 4);
        particleBuffer.SetData(particles);
        frictionMatrixBuffer.SetData(frictionMatrix);

        // Initialize the grid (grid cell size set to rMax)
        InitializeGrid();
        // Allocate grid buffers using gridSize from InitializeGrid()
        gridStartBuffer = new ComputeBuffer(gridSize.x * gridSize.y, sizeof(int));
        gridEndBuffer = new ComputeBuffer(gridSize.x * gridSize.y, sizeof(int));
        particleIndexBuffer = new ComputeBuffer(numParticles, sizeof(uint));

        SetShaderParameters();
    }

    private void Update()
    {
        UpdatePositionsGPU();
        DrawParticles();
    }

    private float[] initRandomMatrixFlat(int numCol)
    {
        float[] forceMatrix = new float[numCol * numCol];
        for (int i = 0; i < numCol * numCol; i++)
        {
            forceMatrix[i] = UnityEngine.Random.Range(-1f, 1f);
        }
        return forceMatrix;
    }

    void InitializeGrid()
    {
        cellSize = rMax;
        gridSize = new Vector2Int(
            Mathf.CeilToInt((screenRight - screenLeft) / cellSize),
            Mathf.CeilToInt((screenTop - screenBottom) / cellSize)
        );
    }

    public void buttonForMatrix()
    {
        frictionMatrix = initRandomMatrixFlat(numColors);
        RespawnParticles();
    }

    private void SpawnParticles(int numPart, int numCol)
    {
        for (int i = 0; i < numPart; i++)
        {
            int colorIndex = Mathf.FloorToInt((i / (float)numPart) * numCol);
            Particle particle = new Particle
            {
                position = new Vector2(
                    UnityEngine.Random.Range(screenLeft, screenRight),
                    UnityEngine.Random.Range(screenBottom, screenTop)
                ),
                velocity = Vector2.zero,
                color = particleColors[colorIndex],
                colorIndex = colorIndex
            };
            particles[i] = particle;
        }
    }

    public void RespawnParticles()
    {
        for (int i = 0; i < numParticles; i++)
        {
            particles[i].position = new Vector2(
                UnityEngine.Random.Range(screenLeft, screenRight),
                UnityEngine.Random.Range(screenBottom, screenTop)
            );
            particles[i].velocity = Vector2.zero;
        }
    }

    private void DrawParticles()
    {
        UpdateParticleMatrices();
        propertyBlock.Clear();

        int batchSize = 1023;
        int batches = batchCache.Length;

        for (int batch = 0; batch < batches; batch++)
        {
            int startIndex = batch * batchSize;
            int count = batchCache[batch].matrices.Length;

            for (int i = 0; i < count; i++)
            {
                batchCache[batch].matrices[i] = particleMatrices[startIndex + i];
                batchCache[batch].colors[i] = particles[startIndex + i].color;
            }

            propertyBlock.SetVectorArray("_Color", batchCache[batch].colors);

            Graphics.DrawMeshInstanced(
                particleMesh,
                0,
                particleMaterial,
                batchCache[batch].matrices,
                count,
                propertyBlock
            );
        }
    }

    private void UpdatePositionsGPU()
    {
        // Update grid buffers so the shader knows which particles are in each cell.
        UpdateGridBuffers();

        frictionMatrixBuffer.SetData(frictionMatrix);
        particleBuffer.SetData(particles);

        int kernelHandle = computeShader.FindKernel("CSMain");
        computeShader.SetBuffer(kernelHandle, "particlesBuffer", particleBuffer);
        computeShader.SetBuffer(kernelHandle, "frictionMatrix", frictionMatrixBuffer);
        computeShader.SetBuffer(kernelHandle, "gridStartBuffer", gridStartBuffer);
        computeShader.SetBuffer(kernelHandle, "gridEndBuffer", gridEndBuffer);
        computeShader.SetBuffer(kernelHandle, "particleIndexBuffer", particleIndexBuffer);

        computeShader.Dispatch(kernelHandle, Mathf.CeilToInt((float)numParticles / 128), 1, 1);



        particleBuffer.GetData(particles);
    }


    private void UpdateGridBuffers()
    {
        int numCells = gridSize.x * gridSize.y;
        // Create temporary lists to collect particle indices for each grid cell.
        List<int>[] cellIndices = new List<int>[numCells];
        for (int i = 0; i < numCells; i++)
        {
            cellIndices[i] = new List<int>();
        }

        // Assign each particle to its corresponding cell based on its position.
        for (int i = 0; i < particles.Length; i++)
        {
            int cellX = Mathf.FloorToInt((particles[i].position.x - screenLeft) / cellSize);
            int cellY = Mathf.FloorToInt((particles[i].position.y - screenBottom) / cellSize);
            // Wrap grid cells (toroidal wrapping) if needed.
            if (cellX < 0) cellX += gridSize.x;
            if (cellX >= gridSize.x) cellX -= gridSize.x;
            if (cellY < 0) cellY += gridSize.y;
            if (cellY >= gridSize.y) cellY -= gridSize.y;
            int cellIndex = cellY * gridSize.x + cellX;
            cellIndices[cellIndex].Add(i);
        }

        // Build arrays for gridStart, gridEnd, and particle indices.
        int[] gridStart = new int[numCells];
        int[] gridEnd = new int[numCells];
        uint[] particleIndices = new uint[particles.Length];
        int currentIndex = 0;
        for (int i = 0; i < numCells; i++)
        {
            gridStart[i] = currentIndex;
            gridEnd[i] = currentIndex + cellIndices[i].Count;
            foreach (int idx in cellIndices[i])
            {
                particleIndices[currentIndex++] = (uint)idx;
            }
        }

        // Update the compute buffers with the new grid data.
        gridStartBuffer.SetData(gridStart);
        gridEndBuffer.SetData(gridEnd);
        particleIndexBuffer.SetData(particleIndices);
    }


    private void OnDestroy()
    {
        if (particleBuffer != null)
            particleBuffer.Dispose();
        if (frictionMatrixBuffer != null)
            frictionMatrixBuffer.Dispose();
        if (gridStartBuffer != null)
            gridStartBuffer.Dispose();
        if (gridEndBuffer != null)
            gridEndBuffer.Dispose();
        if (particleIndexBuffer != null)
            particleIndexBuffer.Dispose();
    }

    void SetShaderParameters()
    {
        computeShader.SetFloat("halfWidthInv", 1f / (width * 0.5f));
        computeShader.SetFloat("halfHeightInv", 1f / (height * 0.5f));
        computeShader.SetFloat("width", width);
        computeShader.SetFloat("height", height);
        computeShader.SetFloat("rMax", rMax);
        computeShader.SetFloat("rMaxSquared", rMax * rMax);
        computeShader.SetFloat("frictionFactor", frictionFactor);
        computeShader.SetFloat("forceFactor", forceFactor);
        computeShader.SetFloat("timeStep", timeStep);
        computeShader.SetFloat("beta", beta);
        computeShader.SetFloat("maxSpeed", maxSpeed);
        computeShader.SetFloat("screenLeft", screenLeft);
        computeShader.SetFloat("screenRight", screenRight);
        computeShader.SetFloat("screenBottom", screenBottom);
        computeShader.SetFloat("screenTop", screenTop);
        computeShader.SetInt("numParticles", numParticles);
        computeShader.SetInt("numColors", numColors);
        computeShader.SetFloat("cellSize", cellSize);
        computeShader.SetInt("gridWidth", gridSize.x);
        computeShader.SetInt("gridHeight", gridSize.y);
    }

    private void UpdateParticleMatrices()
    {
        for (int i = 0; i < particles.Length; i++)
        {
            particleMatrices[i] = Matrix4x4.TRS(
                particles[i].position,
                Quaternion.identity,
                Vector3.one * 0.015f
            );
        }
    }

    private class BatchData
    {
        public Matrix4x4[] matrices;
        public Vector4[] colors;
        public BatchData(int size)
        {
            matrices = new Matrix4x4[size];
            colors = new Vector4[size];
        }
    }
}
