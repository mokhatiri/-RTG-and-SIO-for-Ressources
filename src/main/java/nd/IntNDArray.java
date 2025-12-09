package nd;

import java.util.Arrays;

// Lightweight n-dimensional array wrapper for ints
public class IntNDArray {
    private final int[] shape;
    private final int[] strides;
    private final int[] data;

    public IntNDArray(int... shape) {
        if (shape == null || shape.length == 0) throw new IllegalArgumentException("shape required");
        this.shape = Arrays.copyOf(shape, shape.length);
        this.strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        this.data = new int[stride];
    }

    public static IntNDArray from2D(int[][] arr) {
        int w = arr.length;
        int h = w > 0 ? arr[0].length : 0;
        IntNDArray nd = new IntNDArray(w, h);
        for (int x = 0; x < w; x++)
            for (int y = 0; y < h; y++)
                nd.set(arr[x][y], x, y);
        return nd;
    }

    public static IntNDArray from3D(int[][][] arr) {
        int d0 = arr.length;
        int d1 = d0 > 0 ? arr[0].length : 0;
        int d2 = (d1 > 0) ? arr[0][0].length : 0;
        IntNDArray nd = new IntNDArray(d0, d1, d2);
        for (int i = 0; i < d0; i++)
            for (int j = 0; j < d1; j++)
                for (int k = 0; k < d2; k++)
                    nd.set(arr[i][j][k], i, j, k);
        return nd;
    }

    public int[] shape() { return Arrays.copyOf(shape, shape.length); }

    public int linearIndex(int... idx) {
        if (idx.length != shape.length) throw new IllegalArgumentException("index mismatch");
        int li = 0;
        for (int i = 0; i < idx.length; i++) {
            if (idx[i] < 0 || idx[i] >= shape[i]) throw new IndexOutOfBoundsException("index out of range");
            li += idx[i] * strides[i];
        }
        return li;
    }

    public int get(int... idx) { return data[linearIndex(idx)]; }
    public void set(int val, int... idx) { data[linearIndex(idx)] = val; }
    public int totalSize() { return data.length; }

    public int[] linearToCoords(int index) {
        if (index < 0 || index >= data.length) throw new IndexOutOfBoundsException();
        int[] coords = new int[shape.length];
        for (int i = 0; i < shape.length; i++) {
            coords[i] = (index / strides[i]) % shape[i];
        }
        return coords;
    }
}
