package terrain;

import java.util.Random;

public class NaturalResourceRandomizer {

    private final int width;
    private final int height;
    private final Random rand;

    // Resource type indices
    public static final int SEDIMENTARY_ROCK = 0;
    public static final int GEMSTONES = 1;
    public static final int IRON_ORE = 2;
    public static final int COAL = 3;
    public static final int GOLD_ORE = 4;
    public static final int WOOD = 5;
    public static final int CATTLE_HERD = 6;
    public static final int WOLF_PACK = 7;
    public static final int FISH_SCHOOL = 8;

    public NaturalResourceRandomizer(int width, int height, long seed) {
        this.width = width;
        this.height = height;
        this.rand = new Random(seed);
    }

<<<<<<< HEAD
    public boolean[][] randomizeResourceWeighted(int[][] terrainMap, double[][] flatnessMap, String resourceType, double resourceSpecificProbability) {
        boolean[][] resourceMap = new boolean[width][height];
=======
    public boolean[][][] randomizeResourceWeighted(int[][] terrainMap, double[][] flatnessMap, double[] baseProbabilities) {
        boolean[][][] resourceMap = new boolean[width][height][baseProbabilities.length];
>>>>>>> 454c16b903d3984b6dbc75a4cc6bb3c702486a3b

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int terrain = terrainMap[x][y];
                double flat = flatnessMap[x][y];

                for (int i = 0; i < baseProbabilities.length; i++) {
                    double adjustedProbability = baseProbabilities[i];

<<<<<<< HEAD
                // Combine with randomness
                double chance = resourceSpecificProbability * suitability;
                if (rand.nextDouble() < chance) {
                    resourceMap[x][y] = true;
=======
                    switch (i) {
                        case SEDIMENTARY_ROCK: // prefers water and lowland plains
                            adjustedProbability *= (terrain == 0 || terrain == 1) ? 1.6 : 0.4;
                            break;
                        case GEMSTONES: // prefers mountains and rocky, low-flatness areas
                            adjustedProbability *= (terrain == 2) ? 2.0 : 0.3;
                            adjustedProbability *= (flat < 0.4) ? 1.25 : 0.9;
                            break;
                        case IRON_ORE: // prefers hills and mountains
                            adjustedProbability *= (terrain == 1 || terrain == 2) ? 1.7 : 0.35;
                            break;
                        case COAL: // prefers hills and older sediment areas
                            adjustedProbability *= (terrain == 1) ? 1.5 : (terrain == 2 ? 1.1 : 0.2);
                            break;
                        case GOLD_ORE: // rare, concentrated in rugged mountains
                            adjustedProbability *= (terrain == 2 && flat < 0.25) ? 2.5 : 0.05;
                            break;
                        case WOOD: // prefers forested plains and gentle hills (higher flatness)
                            adjustedProbability *= (terrain == 0) ? 0.3 : (terrain == 1 ? 1.6 : 0.6);
                            adjustedProbability *= (flat > 0.5) ? 1.3 : 0.9;
                            break;
                        case CATTLE_HERD: // domesticated herds prefer open flat plains and pasture (high flatness)
                            adjustedProbability *= (terrain == 0 || terrain == 1) ? 1.6 : 0.25;
                            adjustedProbability *= (flat > 0.6) ? 1.4 : 0.8;
                            break;
                        case WOLF_PACK: // prefers mixed terrain near forests and hills
                            adjustedProbability *= (terrain == 1 || terrain == 2) ? 1.3 : 0.4;
                            break;
                        case FISH_SCHOOL: // only in water
                            adjustedProbability *= (terrain == 0) ? 2.0 : 0.0;
                            break;
                        default:
                            // leave base probability
                            break;
                    }

                    // clamp probability
                    if (Double.isNaN(adjustedProbability) || Double.isInfinite(adjustedProbability)) {
                        adjustedProbability = 0;
                    }
                    adjustedProbability = Math.max(0.0, Math.min(1.0, adjustedProbability));

                    resourceMap[x][y][i] = rand.nextDouble() < adjustedProbability;
>>>>>>> 454c16b903d3984b6dbc75a4cc6bb3c702486a3b
                }
            }
        }

        return resourceMap;
    }
}
