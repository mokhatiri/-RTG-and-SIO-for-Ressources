# RTG and SIO for Resources

Random Terrain Generation (RTG) and Swarm Intelligence Optimization (SIO) visualization using JavaFX.

This project demonstrates terrain/noise generation and Particle Swarm Optimization (PSO) for natural resource placement. It is implemented in Java and uses JavaFX for interactive 2D previews and real-time parameter tuning.

Features
- 2D terrain generation using OpenSimplex / OpenSimplex2-style noise with octave support and normalization
- n-D data types (DoubleNDArray, IntNDArray) and ND-capable algorithm variants for future extension
- Terrain analysis (slope, flatness) and terrain categorization with optional smooth transition zones and neighborhood smoothing
- Natural resource randomization using multi-scale noise veins, terrain modulation, and flatness filters
- Swarm optimization (PSO) to find high-quality resource placement using composite fitness functions
- JavaFX preview UI: NoisePreview2D and PSOPreview for interactive exploration

Quick start
1. Build and run with Maven (example for Windows):

   mvn javafx:run -Djavafx.platform=win

2. Open the application (app.Launcher) to access the previews and controls.

Project structure (important parts)
- src/main/java/app — JavaFX launcher and preview views (Launcher.java, NoisePreview2D.java, PSOPreview.java)
- src/main/java/terrain — Noise.java (OpenSimplex/OpenSimplex2 variants), NoiseMapGenerator.java, TerrainAnalyzer.java, NaturalResourceRandomizer.java
- src/main/java/nd — DoubleNDArray.java, IntNDArray.java, PositionMapper.java
- src/main/java/swarm — PSO implementation and fitness classes (Particle.java, SwarmOptimizer.java, FitnessFunction implementations, ResourcePlacementSolution.java)
- src/main/java/data — ResourceRegistry and resource type definitions
- src/main/resources — UI styles

Key concepts
- Noise generation: OpenSimplex/OpenSimplex2s-style algorithms with skew/unskew transforms, gradient tables and attenuation-based contributions. Multiple octaves (lacunarity/persistence) produce layered detail; outputs are normalized to [0,1].

- Terrain analysis: Compute per-cell slope using central differences, derive flatness = 1 - (slope / maxSlope), and categorize cells into water/plains/hills/mountains. Optional transition zones and neighborhood voting smooth biome boundaries.

- Resource randomization: Uses precomputed coarse noise maps per resource to create veins. Placement decisions are based on stacked thresholds combining noise value, terrain suitability, and flatness, with tunable probabilities per resource.

- Swarm optimization: PSO particles represent candidate placements; velocity and position updates use inertia (w), cognitive (c1) and social (c2) terms. Fitness functions (TerrainFitness, ResourceFitness, MixedFitnessFunction) evaluate locations; the swarm converges to high-quality placement clusters.

Extensibility
- ND arrays and PositionMapper allow generalizing algorithms to n dimensions while keeping 2D views backward-compatible.
- Add new fitness functions or resource types by extending the fitness interfaces and ResourceRegistry.

Where to look in code
- Noise implementations: src/main/java/terrain/Noise.java and NoiseMapGenerator.java
- Terrain analysis: src/main/java/terrain/TerrainAnalyzer.java
- Resource generation: src/main/java/terrain/NaturalResourceRandomizer.java
- PSO and fitness: src/main/java/swarm/*
- JavaFX previews and controls: src/main/java/app/*

Contributors
- @mokhatiri
- @lamseey

License
- (No license specified in the report. Add a LICENSE file or update README to include the preferred license.)

Notes
- The current visualization and controls target a 2D preview. Core data structures and algorithms include n-dimensional support for possible future expansion.
- See Report.md for a more detailed technical explanation and diagrams.
