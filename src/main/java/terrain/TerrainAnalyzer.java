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

    
    // Categorize terrain into types (water, plains, hill, mountain)
    // based on height thresholds.
    public int[][] categorizeTerrain(double waterLevel, double hillLevel, double mountainLevel) {
        int[][] terrain = new int[width][height];

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                double h = heightMap[x][y];
                if (h < waterLevel)
                    terrain[x][y] = 0; // water
                else if (h < hillLevel)
                    terrain[x][y] = 1; // plains
                else if (h < mountainLevel)
                    terrain[x][y] = 2; // hill
                else
                    terrain[x][y] = 3; // mountain
            }
        }

        return terrain;
    }
}
