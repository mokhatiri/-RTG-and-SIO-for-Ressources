package terrain;

import java.util.Random;

public class NaturalResourceRandomizer {

    private final int width;
    private final int height;
    private final Random rand;

    public NaturalResourceRandomizer(int width, int height, long seed) {
        this.width = width;
        this.height = height;
        this.rand = new Random(seed);
    }

    public boolean[][] randomizeResourceWeighted(int[][] terrainMap, double[][] flatnessMap, String resourceType, double resourceSpecificProbability) {
        boolean[][] resourceMap = new boolean[width][height];

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int terrain = terrainMap[x][y];
                double flat = flatnessMap[x][y];
                double suitability = 0.0;

                // Compute suitability a score
                switch(resourceType.toLowerCase()) {
                    case "ore":
                        if (terrain == 2 || terrain == 3) suitability = 1.0;
                        break;
                    case "fertile":
                        if (terrain == 1) suitability = Math.pow(flat, 0.5); // flatter = more suitable
                        break;
                    case "forest":
                        if (terrain == 1 || terrain == 2) suitability = 0.5 + 0.5 * flat;
                        break;
                }

                // Combine with randomness
                double chance = resourceSpecificProbability * suitability;
                if (rand.nextDouble() < chance) {
                    resourceMap[x][y] = true;
                }
            }
        }

        return resourceMap;
    }
}
