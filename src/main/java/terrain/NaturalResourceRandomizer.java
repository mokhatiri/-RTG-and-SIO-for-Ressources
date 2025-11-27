package terrain;

import java.util.Random;

import terrain.RessourceRandomizer.ResourceThresholdModulator;

// Generates natural resource distributions using noise-based vein clustering
// Combines multi-scale noise patterns with terrain preferences for realistic resource placement
public class NaturalResourceRandomizer {

    private static final int NUM_RESOURCES = 9;

    private final Random rand;
    private final double[][][] veinMaps;

    public static final int SEDIMENTARY_ROCK = 0;
    public static final int GEMSTONES = 1;
    public static final int IRON_ORE = 2;
    public static final int COAL = 3;
    public static final int GOLD_ORE = 4;
    public static final int WOOD = 5;
    public static final int CATTLE_HERD = 6;
    public static final int WOLF_PACK = 7;
    public static final int FISH_SCHOOL = 8;

    public NaturalResourceRandomizer(long seed, int mapWidth, int mapHeight, double scale, double offsetX, double offsetY) {
        this.rand = new Random(seed);
        this.veinMaps = new double[NUM_RESOURCES][][];

        // Initialize vein maps for each resource type
        initializeVeinMaps(mapWidth, mapHeight, seed, scale, offsetX, offsetY);
    }

    // Generate coarse noise maps for efficient vein-based distribution
    private void initializeVeinMaps(int width, int height, long baseSeed, double scale, double offsetX, double offsetY) {
        for (int resourceType = 0; resourceType < NUM_RESOURCES; resourceType++) {
            // Unique seed for each resource type to ensure different vein patterns
            long resourceSeed = baseSeed + resourceType * 12345L;
            NoiseMapGenerator noiseGenerator = new NoiseMapGenerator(resourceSeed);
            veinMaps[resourceType] = noiseGenerator.generateWithOffset(width, height, scale*10, 1, 0, 0, 1, offsetX, offsetY);
    }
    }

    public boolean[][][] randomizeResourceWeighted(int[][] terrainMap, double[][] flatnessMap, double[] baseProbabilities) {
        boolean[][][] resourceMap = new boolean[terrainMap.length][terrainMap[0].length][baseProbabilities.length];

        for (int x = 0; x < resourceMap.length; x++) {
            for (int y = 0; y < resourceMap[0].length; y++) {
                int terrain = terrainMap[x][y];
                double flatness = flatnessMap[x][y];

                for (int resourceType = 0; resourceType < baseProbabilities.length; resourceType++) {
                    boolean shouldSpawn = evaluateResourcePlacementStacked(
                            resourceType, veinMaps[resourceType][x][y], terrain, flatness, baseProbabilities[resourceType]
                    );

                    resourceMap[x][y][resourceType] = shouldSpawn;
                }
            }
        }

        return resourceMap;
    }

    private boolean evaluateResourcePlacementStacked(int resourceType, double noiseValue, int terrain, 
                                                      double flatness, double baseProbability) {

        if (baseProbability <= 0.0) {
            return false;
        }

        if (resourceType < 0 || resourceType >= veinMaps.length) {
            return rand.nextDouble() < baseProbability; // Fallback to pure probability if invalid resource type
        }

        double terrainFavoritism = ResourceThresholdModulator.modulateThreshold(resourceType, terrain, flatness, 1 - baseProbability);
        
        if (noiseValue < terrainFavoritism) {
            return false;
        }

        return true;
    }
}
