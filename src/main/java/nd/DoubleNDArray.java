package nd;

import java.util.Arrays;

// Lightweight n-dimensional array wrapper for doubles (flat backing array)
public class DoubleNDArray {
    private final int[] shape;
    private final int[] strides;
    private final double[] data;

    public DoubleNDArray(int... shape) {
        if (shape == null || shape.length == 0) throw new IllegalArgumentException("shape required");
        this.shape = Arrays.copyOf(shape, shape.length);
        this.strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        this.data = new double[stride];
    }

    // Factory to build from a 2D array
    public static DoubleNDArray from2D(double[][] arr) {
        int w = arr.length;
        int h = w > 0 ? arr[0].length : 0;
        DoubleNDArray nd = new DoubleNDArray(w, h);
        for (int x = 0; x < w; x++)
            for (int y = 0; y < h; y++)
                nd.set(arr[x][y], x, y);
        return nd;
    }

    // Factory to build from a 1D array of known shape
    public DoubleNDArray(double[] raw, int... shape) {
        this.shape = Arrays.copyOf(shape, shape.length);
        this.strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        if (raw.length != stride) throw new IllegalArgumentException("raw length mismatch");
        this.data = Arrays.copyOf(raw, raw.length);
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

    public double get(int... idx) {
        return data[linearIndex(idx)];
    }

    public void set(double val, int... idx) {
        data[linearIndex(idx)] = val;
    }

    public int totalSize() { return data.length; }

    // Convert linear index to coordinates
    public int[] linearToCoords(int index) {
        if (index < 0 || index >= data.length) throw new IndexOutOfBoundsException();
        int[] coords = new int[shape.length];
        for (int i = 0; i < shape.length; i++) {
            coords[i] = (index / strides[i]) % shape[i];
        }
        return coords;
    }

    // Provide copy of data for external inspection
    public double[] getRawDataCopy() {
        return Arrays.copyOf(data, data.length);
    }
}
