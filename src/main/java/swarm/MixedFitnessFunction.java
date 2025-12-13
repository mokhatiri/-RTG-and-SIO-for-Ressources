package swarm;

public class MixedFitnessFunction implements FitnessFunction {
    private double resourceCoeff;
    private double terrainCoeff;
    private FitnessFunction resourceFitnessFunction;
    private FitnessFunction terrainFitnessFunction;

    public MixedFitnessFunction(double resourceCoeff, double terrainCoeff, FitnessFunction resourceFitnessFunction, FitnessFunction terrainFitnessFunction) {
        this.resourceCoeff = resourceCoeff;
        this.terrainCoeff = terrainCoeff;
        this.resourceFitnessFunction = resourceFitnessFunction;
        this.terrainFitnessFunction = terrainFitnessFunction;
    }

    @Override
    public double evaluate(int[] x) {
        double denom = (resourceCoeff + terrainCoeff);
        if (denom == 0.0) {
            return 0.0;
        }
        return (resourceCoeff * resourceFitnessFunction.evaluate(x) + terrainCoeff * terrainFitnessFunction.evaluate(x)) / denom;
    }
}
