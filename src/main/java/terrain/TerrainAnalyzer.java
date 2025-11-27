package terrain;

// Analyzes a 2D heightmap to compute terrain metrics such as slope and flatness.
public class TerrainAnalyzer {

    private final double[][] heightMap;
    private final int width;
    private final int height;

    public TerrainAnalyzer(double[][] heightMap) {
        this.heightMap = heightMap;
        this.width = heightMap.length;
        this.height = heightMap[0].length;
    }

    
    // Computes the slope at each point of the heightmap using central differences.
    // return 2D array of slopes
    public double[][] computeSlope() {
        double[][] slope = new double[width][height];

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                double dx = 0;
                double dy = 0;

                if (x > 0 && x < width - 1)
                    dx = (heightMap[x + 1][y] - heightMap[x - 1][y]) / 2.0;
                if (y > 0 && y < height - 1)
                    dy = (heightMap[x][y + 1] - heightMap[x][y - 1]) / 2.0;

                slope[x][y] = Math.sqrt(dx * dx + dy * dy);
            }
        }
        return slope;
    }

    
    // Computes a flatness map (1 = perfectly flat, 0 = very steep)
    // return 2D array of flatness
    public double[][] computeFlatness() {
        double[][] slope = computeSlope();
        double[][] flatness = new double[width][height];
        double maxSlope = 0;
        // find max slope to normalize
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                if (slope[x][y] > maxSlope)
                    maxSlope = slope[x][y];

        if (maxSlope == 0) maxSlope = 1;

        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                flatness[x][y] = 1 - (slope[x][y] / maxSlope);

        return flatness;
    }

    
    // based on height thresholds with neighborhood voting smoothing.
    // First performs height-based categorization, then applies voting to eliminate isolated cells.
    public int[][] categorizeTerrain(double waterLevel, double hillLevel, double mountainLevel, double transition) {
        int[][] terrain = new int[width][height];

        // Step 1: Initial height-based categorization
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                double h = heightMap[x][y];
                if (h < waterLevel)
                    terrain[x][y] = 0;
                else if (waterLevel <= h && h < waterLevel + transition)
                    terrain[x][y] = 2;
                else if (waterLevel + transition <= h && h < hillLevel)
                    terrain[x][y] = 3;
                else if (hillLevel <= h && h < hillLevel + transition)
                    terrain[x][y] = 4;
                else if (hillLevel + transition <= h && h < mountainLevel)
                    terrain[x][y] = 5;
                else if (mountainLevel <= h && h < mountainLevel + transition)
                    terrain[x][y] = 6;
                else
                    terrain[x][y] = 7;
            }
        }

        /*
        0 : WATER
        2 : SHORELINE
        3 : PLAINS
        4 : FOOTHILLS
        5 : HILLS
        6 : MOUNTAIN_BASE
        7 : MOUNTAINS
        */

        // Step 2: Apply neighborhood voting to smooth and eliminate isolated cells
        terrain = applyNeighborhoodVoting(terrain, 3); // 3x3 neighborhood

        return terrain;
    }

    // Each cell is reassigned to the most common terrain type in its neighborhood
    private int[][] applyNeighborhoodVoting(int[][] terrain, int radius) {
        int[][] smoothedTerrain = new int[width][height];

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                // Count votes from neighbors
                int[] votes = new int[8]; // 0-7 terrain types

                for (int dx = -radius / 2; dx <= radius / 2; dx++) {
                    for (int dy = -radius / 2; dy <= radius / 2; dy++) {
                        int nx = x + dx;
                        int ny = y + dy;

                        // Check bounds
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            votes[terrain[nx][ny]]++;
                        }
                    }
                }

                // Find the terrain type with the most votes
                int maxVotes = 0;
                int mostCommonTerrain = terrain[x][y]; // default to current

                for (int terrainType = 0; terrainType < votes.length; terrainType++) {
                    if (votes[terrainType] > maxVotes) {
                        maxVotes = votes[terrainType];
                        mostCommonTerrain = terrainType;
                    }
                }

                smoothedTerrain[x][y] = mostCommonTerrain;
            }
        }

        return smoothedTerrain;
    }
}
