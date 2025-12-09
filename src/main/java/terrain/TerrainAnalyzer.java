package terrain;

import java.util.Arrays;

import nd.DoubleNDArray;
import nd.IntNDArray;

// Analyzes heightmaps (supports 2D and nD via DoubleNDArray) to compute terrain metrics such as slope and flatness.
public class TerrainAnalyzer {

    private final double[][] heightMap; // legacy 2D
    private final DoubleNDArray heightND; // general ND
    private final int width;
    private final int height;

    public TerrainAnalyzer(double[][] heightMap) {
        this.heightMap = heightMap;
        this.width = heightMap.length;
        this.height = heightMap[0].length;
        this.heightND = DoubleNDArray.from2D(heightMap);
    }

    // New constructor for ND-calculations
    public TerrainAnalyzer(DoubleNDArray heightND) {
        this.heightMap = null;
        this.heightND = heightND;
        int[] s = heightND.shape();
        if (s.length == 2) {
            this.width = s[0];
            this.height = s[1];
        } else {
            this.width = s[0];
            this.height = (s.length > 1 ? s[1] : 1);
        }
    }

    
    // Computes the slope at each point of the heightmap using central differences.
    // return 2D array of slopes
    // Legacy 2D helper
    public double[][] computeSlope() {
        DoubleNDArray s = computeSlopeND();
        double[][] out = new double[width][height];
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                out[x][y] = s.get(x, y);
        return out;
    }

    // Compute slope for generic ND heightarray
    public DoubleNDArray computeSlopeND() {
        DoubleNDArray height = heightND;
        int[] shape = height.shape();
        DoubleNDArray slope = new DoubleNDArray(shape);
        int total = slope.totalSize();

        int dims = shape.length;
        int[] coords = new int[dims];
        for (int li = 0; li < total; li++) {
            coords = height.linearToCoords(li);
            double sum2 = 0.0;
            for (int axis = 0; axis < dims; axis++) {
                double deriv = 0.0;
                if (coords[axis] > 0 && coords[axis] < shape[axis] - 1) {
                    int[] prev = Arrays.copyOf(coords, dims);
                    int[] next = Arrays.copyOf(coords, dims);
                    prev[axis] = coords[axis] - 1;
                    next[axis] = coords[axis] + 1;
                    double pv = height.get(prev);
                    double nv = height.get(next);
                    deriv = (nv - pv) / 2.0;
                }
                sum2 += deriv * deriv;
            }
            slope.set(Math.sqrt(sum2), coords);
        }
        return slope;
    }

    
    // Computes a flatness map (1 = perfectly flat, 0 = very steep)
    // return 2D array of flatness
    public double[][] computeFlatness() {
        DoubleNDArray slopeND = computeSlopeND();
        double[][] flatness = new double[width][height];
        double maxSlope = 0;
        // find max slope to normalize
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++) {
                double sVal = slopeND.get(x, y);
                if (sVal > maxSlope) maxSlope = sVal;
            }

        if (maxSlope == 0) maxSlope = 1;

        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                flatness[x][y] = 1 - (slopeND.get(x, y) / maxSlope);

        return flatness;
    }

    
    // based on height thresholds with neighborhood voting smoothing.
    // First performs height-based categorization, then applies voting to eliminate isolated cells.
    public int[][] categorizeTerrain(double waterLevel, double hillLevel, double mountainLevel, double transition) {
        // legacy 2D wrapper for ND implementation
        IntNDArray terrainND = categorizeTerrainND(waterLevel, hillLevel, mountainLevel, transition);
        int[][] terrain = new int[width][height];
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                terrain[x][y] = terrainND.get(x, y);

        // Step 1: Initial height-based categorization
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                double h = (heightMap != null ? heightMap[x][y] : heightND.get(x, y));
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

    // New ND categorization method that returns an IntNDArray
    public nd.IntNDArray categorizeTerrainND(double waterLevel, double hillLevel, double mountainLevel, double transition) {
        DoubleNDArray height = heightND;
        int[] shape = height.shape();
        IntNDArray terrain = new IntNDArray(shape);
        int total = terrain.totalSize();
        for (int li = 0; li < total; li++) {
            int[] coords = terrain.linearToCoords(li);
            double h = height.get(coords);
            int val;
            if (h < waterLevel)
                val = 0;
            else if (waterLevel <= h && h < waterLevel + transition)
                val = 2;
            else if (waterLevel + transition <= h && h < hillLevel)
                val = 3;
            else if (hillLevel <= h && h < hillLevel + transition)
                val = 4;
            else if (hillLevel + transition <= h && h < mountainLevel)
                val = 5;
            else if (mountainLevel <= h && h < mountainLevel + transition)
                val = 6;
            else
                val = 7;
            terrain.set(val, coords);
        }

        // Neighborhood voting smoothing for ND (currently using radius 3 hypercube)
        // We'll perform a single pass to keep behavior consistent
        int[] s = shape;
        IntNDArray smoothed = new IntNDArray(s);
        for (int li = 0; li < total; li++) {
            int[] coords = terrain.linearToCoords(li);
            int[] votes = new int[8];
            int rad = 3;
            // iterate neighbors via recursion
            accumulateVotesND(terrain, coords, votes, rad);
            int maxVotes = 0;
            int mostCommon = terrain.get(coords);
            for (int t = 0; t < votes.length; t++) {
                if (votes[t] > maxVotes) {
                    maxVotes = votes[t];
                    mostCommon = t;
                }
            }
            smoothed.set(mostCommon, coords);
        }

        return smoothed;
    }

    private void accumulateVotesND(IntNDArray terrain, int[] center, int[] votes, int radius) {
        // recursive approach to iterate offsets
        accumulateVotesRec(terrain, center, votes, radius, new int[center.length], 0);
    }

    private void accumulateVotesRec(IntNDArray terrain, int[] center, int[] votes, int radius, int[] offset, int axis) {
        if (axis == center.length) {
            // compute neighbor coords
            int[] neighbor = new int[center.length];
            for (int i = 0; i < center.length; i++) {
                neighbor[i] = center[i] + offset[i] - radius / 2;
            }
            // bounds check
            for (int i = 0; i < neighbor.length; i++) {
                if (neighbor[i] < 0 || neighbor[i] >= terrain.shape()[i]) return;
            }
            int val = terrain.get(neighbor);
            votes[val]++;
            return;
        }
        for (int d = 0; d < radius; d++) {
            offset[axis] = d;
            accumulateVotesRec(terrain, center, votes, radius, offset, axis + 1);
        }
    }
}
