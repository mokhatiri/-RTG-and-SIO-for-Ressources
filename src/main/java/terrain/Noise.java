package terrain;

/**
 * 2D OpenSimplex2-style Noise (simplified version)
 * Adapted from KdotJPG's OpenSimplex2:
 * https://github.com/KdotJPG/OpenSimplex2
 *
 * Produces smooth, pseudo-random 2D noise.
 */
public class Noise {

    // Stretch and squish constants for 2D
    // These convert between grid (integer) coordinates and simplex space
    private static final double STRETCH_CONSTANT = -0.211324865405187; // (1/√2 - 1)/2
    private static final double SQUISH_CONSTANT = 0.366025403784439;  // (√2 - 1)/2

    private static final long DEFAULT_SEED = 0;

    // Permutation table used to pseudo-randomly assign gradients to lattice corners
    private short[] perm;

    // Default constructor uses seed = 0    
    public Noise() {
        this(DEFAULT_SEED);
    }

    
    // Constructor with custom seed.
    // Generates a permutation table based on the seed.
    public Noise(long seed) {
        perm = new short[256];
        short[] source = new short[256]; // used for shuffling before having perm

        // Initialize array 0-255
        for (short i = 0; i < 256; i++)
            source[i] = i;

        // Simple LCG PRNG to scramble the seed
        seed = seed * 6364136223846793005L + 1442695040888963407L;
        seed = seed * 6364136223846793005L + 1442695040888963407L;
        seed = seed * 6364136223846793005L + 1442695040888963407L;

        // Shuffle source[] into perm[] using the PRNG
        for (int i = 255; i >= 0; i--) {
            seed = seed * 6364136223846793005L + 1442695040888963407L; // advance LCG
            int r = (int)((seed + 31) % (i + 1));
            if (r < 0) r += (i + 1);

            perm[i] = source[r]; // shuffle
            source[r] = source[i]; // swap away r
        }
    }

    // 2D gradients used at each lattice corner
    private static final double[] gradients2D = new double[] {
            5,  2,  2,  5,
           -5,  2, -2,  5,
            5, -2,  2, -5,
           -5, -2, -2, -5,
    };

    /**
     * Evaluate noise at 2D point (x, y)
     */
    public double eval(double x, double y) {
        // --- 1. Skew input coordinates into simplex space ---
        double stretchOffset = (x + y) * STRETCH_CONSTANT;
        double xs = x + stretchOffset;
        double ys = y + stretchOffset;

        // --- 2. Find the simplex cell containing the point ---
        int xsb = fastFloor(xs);
        int ysb = fastFloor(ys);

        // --- 3. Unskew cell origin back to (x, y) space ---
        double squishOffset = (xsb + ysb) * SQUISH_CONSTANT;
        double dx0 = x - (xsb + squishOffset); // displacement from cell origin
        double dy0 = y - (ysb + squishOffset);

        double value = 0;

        // --- 4. Contribution from first corner (cell origin) ---
        double attn0 = 2 - dx0 * dx0 - dy0 * dy0; // radial falloff
        if (attn0 > 0) {
            attn0 *= attn0;
            value += attn0 * attn0 * extrapolate(xsb, ysb, dx0, dy0);
        }

        // --- 5. Contribution from second corner (x + 1, y) ---
        double dx1 = dx0 - 1 - SQUISH_CONSTANT;
        double dy1 = dy0 - 0 - SQUISH_CONSTANT;
        double attn1 = 2 - dx1 * dx1 - dy1 * dy1;
        if (attn1 > 0) {
            attn1 *= attn1;
            value += attn1 * attn1 * extrapolate(xsb + 1, ysb, dx1, dy1);
        }

        // --- 6. Contribution from third corner (x, y + 1) ---
        double dx2 = dx0 - 0 - SQUISH_CONSTANT;
        double dy2 = dy0 - 1 - SQUISH_CONSTANT;
        double attn2 = 2 - dx2 * dx2 - dy2 * dy2;
        if (attn2 > 0) {
            attn2 *= attn2;
            value += attn2 * attn2 * extrapolate(xsb, ysb + 1, dx2, dy2);
        }

        // --- 7. Scale result to roughly [-1, 1] ---
        // max value from all corner contributions ≈ 47
        return value / 47;
    }

    // Compute dot product between gradient vector at lattice corner and displacement
    private double extrapolate(int xsb, int ysb, double dx, double dy) {
        // Use permutation table to pick gradient index
        int index = (perm[(perm[xsb & 0xFF] + ysb) & 0xFF] & 0x0E);
        return gradients2D[index] * dx + gradients2D[index + 1] * dy;
    }

    // faster way to floor a double to an int
    private static int fastFloor(double x) {
        return x >= 0 ? (int)x : (int)x - 1;
    }
}
