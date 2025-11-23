package swarm;

import java.util.Random;

public class Particle {

    public double x, y; // current position
    public double vx, vy; // velocity

    public double bestX, bestY; // personal best
    public double bestScore; // personal best score

    private final Random rand = new Random();

    public Particle(int width, int height) {
        this.x = rand.nextDouble() * width;
        this.y = rand.nextDouble() * height;

        this.vx = 0;
        this.vy = 0;
        
        this.bestX = x;
        this.bestY = y;
        this.bestScore = Double.NEGATIVE_INFINITY;
    }
}