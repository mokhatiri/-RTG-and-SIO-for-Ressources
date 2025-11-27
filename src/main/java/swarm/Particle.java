package swarm;

public class Particle {
    private double[] position; // Current position of the particle
    private double[] velocity; // Current velocity of the particle
    private double[] personalBestPosition; // Best position found by this particle
    private double personalBestFitness; // Fitness value of the personal best position

    public Particle(int dimensions, double maxPosition, double minPosition) {
        position = new double[dimensions];
        velocity = new double[dimensions];
        personalBestPosition = new double[dimensions];
        personalBestFitness = Double.NEGATIVE_INFINITY;

        // Initialize position and velocity randomly
        for (int i = 0; i < dimensions; i++) {
            position[i] = minPosition + (maxPosition - minPosition) * Math.random();
            velocity[i] = -1.0 + 2.0 * Math.random(); // Random velocity between -1 and 1
            personalBestPosition[i] = position[i]; // Initial personal best is current position
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
}