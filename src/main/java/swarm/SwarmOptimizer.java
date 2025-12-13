package swarm;
// No wildcard function import required

import java.util.Random;
import java.util.ArrayList;
import java.util.List;
import java.util.function.*;

public class SwarmOptimizer {
    // TODO
    private int dimensions;
    private List<Particle> particles = new ArrayList<>();
    private double w;
    private double c1;
    private double c2;
    // evaluate finctness (double a, double b, double[dimensions] x) -> Double 
    private MixedFitnessFunction evaluateFitness;
    private Function<double[], Double> fitnessAdapter; // continuous adapter
    Random rand = new Random();
    private double globalBestFitness = Double.NEGATIVE_INFINITY;
    private double[] globalBestPosition;

    // New constructor: pass resourceFitness and terrainFitness functions and the grid shape to map continuous pos -> discrete
    public SwarmOptimizer(int dimensions, double w, double c1, double c2, double resource_coeff, double terrain_coeff, 
                          FitnessFunction resourceFitness, FitnessFunction terrainFitness, int[] gridShape) {
        this.dimensions = dimensions;
        this.w = w;   // inertia weight
        this.c1 = c1; // cognitive coefficient
        this.c2 = c2; // social coefficient
        // create the mixed function from the passed-in functions
        evaluateFitness = new MixedFitnessFunction(resource_coeff, terrain_coeff, resourceFitness, terrainFitness);
        // adapter for continuous positions
        this.fitnessAdapter = FitnessAdapter.toContinuous(evaluateFitness, gridShape);
        this.globalBestPosition = new double[dimensions];
    }

    public void addParticle(double maxPosition, double minPosition){
        Particle p = new Particle(dimensions, maxPosition, minPosition, this.fitnessAdapter);
        particles.add(p);

        // Initialize global best from initial random population
        double f = p.getPersonalBestFitness();
        if (f > globalBestFitness) {
            globalBestFitness = f;
            System.arraycopy(p.getPersonalBestPosition(), 0, globalBestPosition, 0, globalBestPosition.length);
        }
    }

    public List<Particle> getParticles() {
        return particles;
    }

    public double getGlobalBestFitness() {
        return globalBestFitness;
    }

    public double[] getGlobalBestPosition() {
        return globalBestPosition;
    }

    public void nextIteration(){
        for (Particle particle : particles) {
            double bestFitness = particle.update(w, c1, c2, globalBestPosition);
            if (bestFitness > globalBestFitness){
                globalBestFitness = bestFitness;
                System.arraycopy(particle.getPersonalBestPosition(), 0, globalBestPosition, 0, globalBestPosition.length);
            }
        }
    }

}
