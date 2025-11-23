package swarm;

public class ResourceFitness implements FitnessFunction {

    private final int[][] terrain;
    private final double[][] flatness;

    public ResourceFitness(int[][] terrain, double[][] flatness) {
        this.terrain = terrain;
        this.flatness = flatness;
    }

    @Override
    public double evaluate(int x, int y) {
        if (x < 0 || y < 0 || x >= terrain.length || y >= terrain[0].length)
            return -1000;

        int type = terrain[x][y];
        double flat = flatness[x][y];

        double baseScore = 0;

        // Example: optimizing for "ore"
        if (type == 3 || type == 2) baseScore += 1.0;
        baseScore += flat * 0.3;

        return baseScore;
    }
}
