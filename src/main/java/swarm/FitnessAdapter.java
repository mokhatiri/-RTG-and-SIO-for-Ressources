package swarm;

import java.util.function.Function;

import nd.PositionMapper;

// Adapts a FitnessFunction (int[] coords) to a Function<double[], Double> by rounding and clamping.
public class FitnessAdapter {
    public static Function<double[], Double> toContinuous(FitnessFunction f, int[] shape) {
        return double_pos -> {
            int[] int_pos = PositionMapper.roundAndClamp(double_pos, shape);
            return f.evaluate(int_pos);
        };
    }
}
