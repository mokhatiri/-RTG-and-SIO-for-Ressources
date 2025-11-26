package swarm;

public class TerrainFitness implements FitnessFunction {

    private final int[][] terrain;
    private final double[][] flatness;
    private final double water_value;
    private final double plains_value;
    private final double hills_value;
    private final double mountains_value;
    private final double flatness_coeff;

    public TerrainFitness(double[][] flatness,
                          double flatness_coeff,
                          int[][] terrain,
                          double water_value,
                          double plains_value,
                          double hills_value,
                          double mountains_value) {
        this.terrain = terrain;
        this.flatness = flatness;
        this.water_value = water_value;
        this.plains_value = plains_value;
        this.hills_value = hills_value;
        this.mountains_value = mountains_value;
        this.flatness_coeff = flatness_coeff;
    }

    @Override
    public double evaluate(int x, int y) {
        int t = terrain[x][y];

        double score = switch (t) {
            case 0 -> water_value;  // water
            case 1 -> plains_value;  // plains
            case 2 -> hills_value;  // hills
            case 3 -> mountains_value;  // mountains
            default -> 0;
        };

        // combine with flatness
        return score * (1 - (1  - flatness[x][y]) * flatness_coeff);
    }
}
