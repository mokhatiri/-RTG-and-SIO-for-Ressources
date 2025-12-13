package app.psoPreview;

/**
 * Reduces the 0-7 terrain palette used by the preview renderer into a 0-3 palette
 * expected by {@link swarm.TerrainFitness}:
 * 0=water, 1=plains, 2=hills, 3=mountains.
 */
public final class TerrainTypeReducer {
    private TerrainTypeReducer() {}

    public static int[][] reduceTo4Types(int[][] terrainTypes) {
        int w = terrainTypes.length;
        int h = terrainTypes[0].length;
        int[][] out = new int[w][h];

        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                int t = terrainTypes[x][y];
                // Preview terrain types: 0,1 water; 2 beach; 3 plains; 4 foothills; 5 hills; 6 mountain_base; 7 mountains
                if (t <= 1) {
                    out[x][y] = 0;
                } else if (t == 2 || t == 3) {
                    out[x][y] = 1;
                } else if (t == 4 || t == 5) {
                    out[x][y] = 2;
                } else {
                    out[x][y] = 3;
                }
            }
        }
        return out;
    }
}
