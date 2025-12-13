package swarm;
import java.util.function.Function;

public class Particle {
    private int dimensions;
    private double[] position; // Current position of the particle
    private double[] velocity; // Current velocity of the particle
    private double[] personalBestPosition; // Best position found by this particle
    private double personalBestFitness; // Fitness value of the personal best position
    private Function<double[], Double> evaluateFitness; // Fitness evaluation function
    private final double minPosition;
    private final double maxPosition;

    public Particle(int dimensions, double maxPosition, double minPosition, Function<double[], Double> evaluateFitness) {
        this.dimensions = dimensions;
        this.position = new double[dimensions];
        this.velocity = new double[dimensions];
        this.personalBestPosition = new double[dimensions];
        this.personalBestFitness = Double.NEGATIVE_INFINITY;
        this.evaluateFitness = evaluateFitness;
        this.minPosition = minPosition;
        this.maxPosition = maxPosition;

        // Initialize position and velocity randomly
        for (int i = 0; i < dimensions; i++) {
            this.position[i] = minPosition + (maxPosition - minPosition) * Math.random();
            this.velocity[i] = -1.0 + 2.0 * Math.random(); // Random velocity between -1 and 1
            this.personalBestPosition[i] = position[i]; // Initial personal best is current position
        }
    }

    // Getters and Setters
    public double[] getPosition() {
        return position;
    }

    public void setPosition(double[] position) {
        this.position = position;
    }

    public double[] getVelocity() {
        return velocity;
    }

    public void setVelocity(double[] velocity) {
        this.velocity = velocity;
    }

    public double[] getPersonalBestPosition() {
        return personalBestPosition;
    }

    public void setPersonalBestPosition(double[] personalBestPosition) {
        this.personalBestPosition = personalBestPosition;
    }

    public double getPersonalBestFitness() {
        return personalBestFitness;
    }

    public void setPersonalBestFitness(double personalBestFitness) {
        this.personalBestFitness = personalBestFitness;
    }

    // Update personal best
    public void updatePersonalBest(double currentFitness) {
        if (currentFitness > personalBestFitness) { // For maximization problems
            personalBestFitness = currentFitness;
            System.arraycopy(position, 0, personalBestPosition, 0, position.length);
        }
    }

    public double update(double w, double c1, double c2, double[] globalBestPosition){
        for(int i = 0; i < dimensions; i++ ){
            double r1 = Math.random();
            double r2 = Math.random();

            double gb = (globalBestPosition != null && globalBestPosition.length > i)
                    ? globalBestPosition[i]
                    : position[i];

            velocity[i] = w * velocity[i]
                    + c1 * r1 * (personalBestPosition[i] - position[i])
                    + c2 * r2 * (gb - position[i]);
            position[i] += velocity[i];

            // Keep particles inside bounds: reflect velocity when hitting edges.
            if (position[i] < minPosition) {
                position[i] = minPosition;
                velocity[i] = -velocity[i];
            } else if (position[i] > maxPosition) {
                position[i] = maxPosition;
                velocity[i] = -velocity[i];
            }
        }

        updatePersonalBest(this.evaluateFitness.apply(position));
        return personalBestFitness;
    }

    // Backward compatible overload (treat scalar as "same best" for all axes)
    public double update(double w, double c1, double c2, double globalBest){
        double[] gb = new double[dimensions];
        for (int i = 0; i < dimensions; i++) {
            gb[i] = globalBest;
        }
        return update(w, c1, c2, gb);
    }
}