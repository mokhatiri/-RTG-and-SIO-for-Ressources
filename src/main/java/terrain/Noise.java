package terrain;

public class Noise {

    // Rotation constants for OpenSimplex2S
    private static final double R2 = 0.5 * (Math.sqrt(3.0) - 1.0);
    private static final double R2_INV = (3.0 - Math.sqrt(3.0)) / 6.0;

    // True OpenSimplex2S gradient vectors (24 directions evenly around unit circle)
    private static final double[] grad2 = new double[] {
            0.130526192220052,  0.99144486137381,
            0.382683432365090,  0.923879532511287,
            0.608761429008721,  0.793353340291235,
            0.793353340291235,  0.608761429008721,
            0.923879532511287,  0.382683432365090,
            0.991444861373810,  0.130526192220051,
            0.991444861373810, -0.130526192220051,
            0.923879532511287, -0.382683432365090,
            0.793353340291235, -0.608761429008721,
            0.608761429008721, -0.793353340291235,
            0.382683432365090, -0.923879532511287,
            0.130526192220052, -0.991444861373810,
           -0.130526192220052, -0.991444861373810,
           -0.382683432365090, -0.923879532511287,
           -0.608761429008721, -0.793353340291235,
           -0.793353340291235, -0.608761429008721,
           -0.923879532511287, -0.382683432365090,
           -0.991444861373810, -0.130526192220052,
           -0.991444861373810,  0.130526192220051,
           -0.923879532511287,  0.382683432365090,
           -0.793353340291235,  0.608761429008721,
           -0.608761429008721,  0.793353340291235,
           -0.382683432365090,  0.923879532511287,
           -0.130526192220052,  0.991444861373810
    };

    private short[] perm;

    public Noise(long seed) {
        perm = new short[256];
        short[] source = new short[256];
        for (short i = 0; i < 256; i++) source[i] = i;

        for (int i = 255; i >= 0; i--) {
            seed = seed * 6364136223846793005L + 1442695040888963407L;
            int r = (int)((seed + 31) % (i + 1));
            if (r < 0) r += (i + 1);
            perm[i] = source[r];
            source[r] = source[i];
        }
    }

    public double eval(double x, double y) {

        // --- Rotation into rhombus domain ---
        double s = (x + y) * R2;
        double xs = x + s;
        double ys = y + s;

        int xsb = fastFloor(xs);
        int ysb = fastFloor(ys);

        // --- Unrotate back to (x,y) space ---
        double t = (xsb + ysb) * R2_INV;
        double dx0 = x - (xsb - t);
        double dy0 = y - (ysb - t);

        double value = 0.0;
        
        // We evaluate **4 points** (SuperSimplex)
        value += contribution(xsb, ysb, dx0, dy0);
        value += contribution(xsb + 1, ysb, dx0 - 1 + R2_INV, dy0 + R2_INV);
        value += contribution(xsb, ysb + 1, dx0 + R2_INV, dy0 - 1 + R2_INV);
        value += contribution(xsb + 1, ysb + 1, dx0 - 1 + 2*R2_INV, dy0 - 1 + 2*R2_INV);

        return value * 99.0; // empirical scaling to approx [-1,1]
    }

    private double contribution(int i, int j, double dx, double dy) {

        double attn = 0.5 - dx*dx - dy*dy;
        if (attn <= 0) return 0;

        int index = perm[(perm[i & 255] + j) & 255] & 0x1E; // selects gradient
        double gx = grad2[index];
        double gy = grad2[index + 1];

        attn *= attn;
        return attn * attn * (gx*dx + gy*dy);
    }

    private static int fastFloor(double x) {
        return x >= 0 ? (int)x : (int)x - 1;
    }
}
