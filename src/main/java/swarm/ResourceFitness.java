package swarm;

import nd.DoubleNDArray;
import nd.IntNDArray;

public class ResourceFitness implements FitnessFunction {

    private final DoubleNDArray flatness;
    private final IntNDArray resourceTerrain; // last axis: resource index
    private final double flatness_coeff;

    double sedimentaryRockValue;
    double gemstonesValue;
    double ironOreValue;
    double coalValue;
    double goldOreValue;
    double woodValue;
    double cattleHerdValue;
    double wolfPackValue;
    double fishSchoolValue;

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


    public ResourceFitness(double[][] flatness,
                           double flatness_coeff,
                           int[][][] resourceTerrain,
                           double sedimentaryRockValue,
                           double gemstonesValue,
                           double ironOreValue,
                           double coalValue,
                           double goldOreValue,
                           double woodValue,
                           double cattleHerdValue,
                           double wolfPackValue,
                           double fishSchoolValue) {
        this.flatness = DoubleNDArray.from2D(flatness);
        this.resourceTerrain = IntNDArray.from3D(resourceTerrain);
        this.sedimentaryRockValue = sedimentaryRockValue;
        this.gemstonesValue = gemstonesValue;
        this.ironOreValue = ironOreValue;
        this.coalValue = coalValue;
        this.goldOreValue = goldOreValue;
        this.woodValue = woodValue;
        this.cattleHerdValue = cattleHerdValue;
        this.wolfPackValue = wolfPackValue;
        this.fishSchoolValue = fishSchoolValue;
        this.flatness_coeff = flatness_coeff;
    }

    @Override
    public double evaluate(int[] pos) {
        double score = 0.0;

        // Build a coordinate array including resource axis
        int[] base = pos.clone();
        // for each resource type, probe resourceTerrain at base coords + resource index
        for (int r = 0; r < 9; r++) {
            int[] idx = new int[base.length + 1];
            System.arraycopy(base, 0, idx, 0, base.length);
            idx[base.length] = r;
            if (resourceTerrain.get(idx) == 1) {
                switch (r) {
                    case SEDIMENTARY_ROCK -> score += sedimentaryRockValue;
                    case GEMSTONES -> score += gemstonesValue;
                    case IRON_ORE -> score += ironOreValue;
                    case COAL -> score += coalValue;
                    case GOLD_ORE -> score += goldOreValue;
                    case WOOD -> score += woodValue;
                    case CATTLE_HERD -> score += cattleHerdValue;
                    case WOLF_PACK -> score += wolfPackValue;
                    case FISH_SCHOOL -> score += fishSchoolValue;
                }
            }
        }

        // modify score based on flatness
        return score * (1 - (1 - flatness.get(pos)) * flatness_coeff);
    }
}
