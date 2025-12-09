package terrain;
public class NoiseMapGenerator {
    private final Noise noise;

    public NoiseMapGenerator(long seed) {
        this.noise = new Noise(seed);
    }

    private double layeredNoise(double x, double y, double width, double height, double scale, int octaves, double persistence, double lacunarity, double flattenPower) {
        if (scale <= 0) scale = 0.0001;

        double nx = x / (double) width * scale;
        double ny = y / (double) height * scale;

        double val = 0, freq = 1, amp = 1, max = 0;
        for (int o = 0; o < octaves; o++) {
            
            val += noise.eval(nx * freq, ny * freq) * amp;
            max += amp;
            amp *= persistence;
            freq *= lacunarity;
        }

        double normalized = (val / max + 1) / 2.0;
        double t = (normalized - 0.5) * 2.0;  // shift to [-1, 1]
        t = t / (1.0 + flattenPower * (t * t)); // smoothly compress
        normalized = t * 0.5 + 0.5;


        return Math.min(1.0, Math.max(0.0, normalized));
    }

    public double[][] generate(double width, double height, double scale, int octaves, double persistence, double lacunarity, double flattenPower) {
        double[][] map = new double[(int) width][(int) height];
        for (int x = 0; x < (int) width; x++)
            for (int y = 0; y < (int) height; y++)
                map[x][y] = layeredNoise(x, y, width, height, scale, octaves, persistence, lacunarity, flattenPower);
        return map;
    }

    public double[][] generateWithOffset(double width, double height, double scale, int octaves, double persistence, double lacunarity, double flattenPower, double offsetX, double offsetY) {
        double[][] map = new double[(int) width][(int) height];
        for (int x = 0; x < (int) width; x++)
            for (int y = 0; y < (int) height; y++)
                map[x][y] = layeredNoise(x + offsetX, y + offsetY, width, height, scale, octaves, persistence, lacunarity, flattenPower);
        return map;
    }
}
