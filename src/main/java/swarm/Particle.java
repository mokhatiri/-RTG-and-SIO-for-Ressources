package swarm;
import java.util.function.Function;

public class Particle {
    private int dimensions;
    private double[] position; // Current position of the particle
    private double[] velocity; // Current velocity of the particle
    private double[] personalBestPosition; // Best position found by this particle
    private double personalBestFitness; // Fitness value of the personal best position
    private Function<double[], Double> evaluateFitness; // Fitness evaluation function

    public Particle(int dimensions, double maxPosition, double minPosition, Function<double[], Double> evaluateFitness) {
        this.dimensions = dimensions;
        this.position = new double[dimensions];
        this.velocity = new double[dimensions];
        this.personalBestPosition = new double[dimensions];
        this.personalBestFitness = Double.NEGATIVE_INFINITY;
        this.evaluateFitness = evaluateFitness;

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

    public double update(double w, double c1, double c2, double globalBest){
        for(int i = 0; i < dimensions; i++ ){
            double r1 = Math.random(); // changes for each particle
            double r2 = Math.random(); // changes for each particle
            // Update velocity
            velocity[i] = w * velocity[i]
                    + c1 * r1 * (personalBestPosition[i] - position[i])
                    + c2 * r2 * (globalBest - position[i]);
            // Update position
            position[i] += velocity[i];
        }

        updatePersonalBest(this.evaluateFitness.apply(position));
        return personalBestFitness;
    }
}