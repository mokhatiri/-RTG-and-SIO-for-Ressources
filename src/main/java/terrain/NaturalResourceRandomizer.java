package terrain;

import java.util.Random;

import nd.DoubleNDArray;
import nd.IntNDArray;
import terrain.RessourceRandomizer.ResourceThresholdModulator;

// Generates natural resource distributions using noise-based vein clustering
// Combines multi-scale noise patterns with terrain preferences for realistic resource placement
public class NaturalResourceRandomizer {

    private static final int NUM_RESOURCES = 9;

    private final Random rand;
    private final double[][][] veinMaps;
    // ND-friendly caches
    private final nd.DoubleNDArray[] veinMapsND;

    public static final int SEDIMENTARY_ROCK = 0;
    public static final int GEMSTONES = 1;
    public static final int IRON_ORE = 2;
    public static final int COAL = 3;
    public static final int GOLD_ORE = 4;
    public static final int WOOD = 5;
    public static final int CATTLE_HERD = 6;
    public static final int WOLF_PACK = 7;
    public static final int FISH_SCHOOL = 8;

    public NaturalResourceRandomizer(long seed, double mapWidth, double mapHeight, double scale, double offsetX, double offsetY) {
        this.rand = new Random(seed);
        this.veinMaps = new double[NUM_RESOURCES][][];
        this.veinMapsND = new DoubleNDArray[NUM_RESOURCES];

        // Initialize vein maps for each resource type
        initializeVeinMaps(mapWidth, mapHeight, seed, scale, offsetX, offsetY);
    }

    // Generate coarse noise maps for efficient vein-based distribution
    private void initializeVeinMaps(double width, double height, long baseSeed, double scale, double offsetX, double offsetY) {
        for (int resourceType = 0; resourceType < NUM_RESOURCES; resourceType++) {
            // Unique seed for each resource type to ensure different vein patterns
            long resourceSeed = baseSeed + resourceType * 12345L;
            NoiseMapGenerator noiseGenerator = new NoiseMapGenerator(resourceSeed);
            veinMaps[resourceType] = noiseGenerator.generateWithOffset(width, height, scale*10, 1, 0, 0, 1, offsetX, offsetY);
            veinMapsND[resourceType] = DoubleNDArray.from2D(veinMaps[resourceType]);
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

    // n-dimensional resource placement: returns an IntNDArray with an added last axis of resource types (0/1)
    public IntNDArray randomizeResourceWeightedND(IntNDArray terrainMap, DoubleNDArray flatnessMap, double[] baseProbabilities) {
        int[] shape = terrainMap.shape();
        // output shape: spatial dims + resource axis
        int[] outShape = new int[shape.length + 1];
        System.arraycopy(shape, 0, outShape, 0, shape.length);
        outShape[shape.length] = baseProbabilities.length;
        IntNDArray resourceMapND = new IntNDArray(outShape);

        int total = 1;
        for (int s : shape) total *= s;
        for (int li = 0; li < total; li++) {
            int[] pos = terrainMap.linearToCoords(li);
            int terrain = terrainMap.get(pos);
            // get flatness
            double flatness = flatnessMap.get(pos);
            for (int r = 0; r < baseProbabilities.length; r++) {
                double noiseValue = veinMapsND[r].get(pos);
                boolean shouldSpawn = evaluateResourcePlacementStackedND(r, noiseValue, terrain, flatness, baseProbabilities[r]);
                int[] outIdx = new int[pos.length + 1];
                System.arraycopy(pos, 0, outIdx, 0, pos.length);
                outIdx[pos.length] = r;
                resourceMapND.set(shouldSpawn ? 1 : 0, outIdx);
            }
        }
        return resourceMapND;
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

    private boolean evaluateResourcePlacementStackedND(int resourceType, double noiseValue, int terrain,
                                                       double flatness, double baseProbability) {
        return evaluateResourcePlacementStacked(resourceType, noiseValue, terrain, flatness, baseProbability);
    }
}
