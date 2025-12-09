package swarm;

// Represents a potentially n-dimensional placement solution; now holds arbitrary coordinates
public class ResourcePlacementSolution {
    public final int[] coords;

    public ResourcePlacementSolution(int... coords) {
        this.coords = coords;
    }
}
