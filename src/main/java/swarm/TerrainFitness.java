package swarm;

public class TerrainFitness implements FitnessFunction {

    private final int[][] terrain;
    private final double[][] flatness;

    public TerrainFitness(int[][] terrain, double[][] flatness) {
        this.terrain = terrain;
        this.flatness = flatness;
    }

    @Override
    public double evaluate(int x, int y) {
        int t = terrain[x][y];

        // base score from terrain
        double score = switch (t) {
            case 0 -> 0.1;  // water
            case 1 -> 0.6;  // plains
            case 2 -> 0.8;  // hills
            case 3 -> 0.9;  // mountains
            default -> 0;
        };

        // combine with flatness
        return score * (0.5 + 0.5 * flatness[x][y]);
    }
}
