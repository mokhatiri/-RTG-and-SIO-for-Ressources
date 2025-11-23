package terrain;

/**
 * Generates 2D noise maps using OpenSimplexNoise with support for octaves.
 */
public class NoiseMapGenerator {

    private final Noise noise;

    public NoiseMapGenerator(long seed) {
        this.noise = new Noise(seed);
    }

    public double[][] generate(
            int width,
            int height,
            double scale,
            int octaves,
            double persistence,
            double lacunarity
    ) {
        double[][] map = new double[width][height];

        if (scale <= 0) scale = 0.0001; // prevent division by zero

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {

                double amplitude = 1;
                double frequency = 1;
                double noiseHeight = 0;

                // Apply multiple octaves
                for (int o = 0; o < octaves; o++) {
                    double sampleX = x / scale * frequency;
                    double sampleY = y / scale * frequency;

                    double value = noise.eval(sampleX, sampleY);
                    noiseHeight += value * amplitude;

                    amplitude *= persistence;
                    frequency *= lacunarity;
                }

                // Normalize to 0..1
                map[x][y] = (noiseHeight + 1) / 2.0;
            }
        }

        return map;
    }
}
