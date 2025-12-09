package nd;

public class PositionMapper {
    public static int[] roundAndClamp(double[] pos, int[] shape) {
        int dims = Math.min(pos.length, shape.length);
        int[] coords = new int[shape.length];
        for (int i = 0; i < shape.length; i++) {
            if (i < dims) {
                int c = (int) Math.round(pos[i]);
                if (c < 0) c = 0;
                if (c >= shape[i]) c = shape[i] - 1;
                coords[i] = c;
            } else {
                coords[i] = 0;
            }
        }
        return coords;
    }
}
