package swarm;

public class ResourceFitness implements FitnessFunction {

    private final double[][] flatness;
    private final int[][][] resourceTerrain;
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
        this.flatness = flatness;
        this.resourceTerrain = resourceTerrain;
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
    public double evaluate(int x, int y) {
        double score = 0.0;

        if (resourceTerrain[x][y][SEDIMENTARY_ROCK] == 1)
            score += sedimentaryRockValue;
        if (resourceTerrain[x][y][GEMSTONES] == 1)
            score += gemstonesValue;
        if (resourceTerrain[x][y][IRON_ORE] == 1)
            score += ironOreValue;
        if (resourceTerrain[x][y][COAL] == 1)
            score += coalValue;
        if (resourceTerrain[x][y][GOLD_ORE] == 1)
            score += goldOreValue;
        if (resourceTerrain[x][y][WOOD] == 1)
            score += woodValue;
        if (resourceTerrain[x][y][CATTLE_HERD] == 1)
            score += cattleHerdValue;
        if (resourceTerrain[x][y][WOLF_PACK] == 1)
            score += wolfPackValue;
        if (resourceTerrain[x][y][FISH_SCHOOL] == 1)
            score += fishSchoolValue;

        // modify score based on flatness
        return score * (1 - (1  - flatness[x][y]) * flatness_coeff);
    }
}
