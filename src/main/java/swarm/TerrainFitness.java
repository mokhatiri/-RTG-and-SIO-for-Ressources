package swarm;

import nd.DoubleNDArray;
import nd.IntNDArray;

public class TerrainFitness implements FitnessFunction {

    private final IntNDArray terrain;
    private final DoubleNDArray flatness;
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
        this.terrain = IntNDArray.from2D(terrain);
        this.flatness = DoubleNDArray.from2D(flatness);
        this.water_value = water_value;
        this.plains_value = plains_value;
        this.hills_value = hills_value;
        this.mountains_value = mountains_value;
        this.flatness_coeff = flatness_coeff;
    }

    @Override
    public double evaluate(int[] pos) {
        int t = terrain.get(pos);
        double score;
        switch (t) {
            case 0: score = water_value; break;
            case 1: score = plains_value; break;
            case 2: score = hills_value; break;
            case 3: score = mountains_value; break;
            default: score = 0; break;
        }
        return score * (1 - (1 - flatness.get(pos)) * flatness_coeff);
    }
}
