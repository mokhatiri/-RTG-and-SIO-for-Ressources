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
    }

    public void addParticle(double maxPosition, double minPosition){
        particles.add(new Particle(dimensions, maxPosition, minPosition, this.fitnessAdapter));
    }

    public List<Particle> getParticles() {
        return particles;
    }

    public void nextIteration(){
        double bestFitness;
        for (Particle particle : particles) {
            for (int i = 0; i < dimensions; i++) {
                bestFitness = particle.update(w, c1, c2, globalBestFitness);
                if (bestFitness > globalBestFitness){
                    globalBestFitness = bestFitness;
                }
            }
        }
    }

}
