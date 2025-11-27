package swarm;

import java.util.Random;
import java.util.ArrayList;
import java.util.List;

public class SwarmOptimizer {
    // TODO
    private int dimensions;
    private List<Particle> particles = new ArrayList<>();
    private double r1;
    private double r2;
    private double w;
    private double c1;
    private double c2;
    Random rand = new Random();
    private double globalBestFitness = Double.NEGATIVE_INFINITY;

    public SwarmOptimizer(int dimensions, double w, double c1, double c2) {
        this.dimensions = dimensions;
        this.w = w;
        this.c1 = c1;
        this.c2 = c2;
        this.r1 = rand.nextDouble();
        this.r2 = rand.nextDouble();
    }

    public void addParticle(double maxPosition, double minPosition){
        particles.add(new Particle(dimensions, maxPosition, minPosition));
    }

    public List<Particle> getParticles() {
        return particles;
    }

    public void nextIteration(){
        for (Particle particle : particles) {
            // Update particle position based on its velocity
            double[] position = particle.getPosition();
            double[] velocity = particle.getVelocity();
            for (int i = 0; i < dimensions; i++) {
                position[i] += velocity[i];
                velocity[i] = w * velocity[i]
                        + c1 * r1 * (particle.getPersonalBestPosition()[i] - position[i])
                        + c2 * r2 * (/*globalBestPosition[i]*/ 0 - position[i]); // TODO: global best
            }
            particle.setPosition(position);
        }
    }

}
