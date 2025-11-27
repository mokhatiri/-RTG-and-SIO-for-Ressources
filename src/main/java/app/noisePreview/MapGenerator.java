package app.noisePreview;

import terrain.NoiseMapGenerator;
import terrain.TerrainAnalyzer;
import terrain.NaturalResourceRandomizer;

// Handles generation of terrain maps (height, terrain types, flatness) and resource maps
public class MapGenerator {
    private static final int WIDTH = 128;
    private static final int HEIGHT = 128;

    private double[][] heightMap;
    private int[][] terrainTypes;
    private double[][] flatnessMap;
    private boolean[][][] resourceMaps;

    public void generateTerrainMaps(TerrainGenerationParams params, double offsetX, double offsetY) {
        NoiseMapGenerator generator = new NoiseMapGenerator(params.seed);
        heightMap = generator.generateWithOffset(
                WIDTH, HEIGHT,
                params.scale, params.octaves, params.persistence, params.lacunarity, params.flattenPower,
                offsetX, offsetY
        );
        
        TerrainAnalyzer analyzer = new TerrainAnalyzer(heightMap);
        terrainTypes = analyzer.categorizeTerrain(
                params.waterLevel, params.hillLevel, params.mountainLevel, params.transition
        );
        flatnessMap = analyzer.computeFlatness();
    }

    public void generateResourceMaps(ResourceGenerationParams params, double offsetX, double offsetY) {
        NaturalResourceRandomizer resourceRandomizer = new NaturalResourceRandomizer(params.randomizerSeed, WIDTH, HEIGHT, params.scale, offsetX, offsetY);
        resourceMaps = resourceRandomizer.randomizeResourceWeighted(
                                    terrainTypes,
                                    flatnessMap,
                                    params.getProbabilitiesArray());
    }

    // Getters
    public double[][] getHeightMap() {
        return heightMap;
    }

    public int[][] getTerrainTypes() {
        return terrainTypes;
    }

    public double[][] getFlatnessMap() {
        return flatnessMap;
    }

    public boolean[][][] getResourceMaps() {
        return resourceMaps;
    }
}
