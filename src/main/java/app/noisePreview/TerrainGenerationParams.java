package app.noisePreview;

// Encapsulates terrain generation parameters for noise and categorization
public class TerrainGenerationParams {
    public double scale = 2.0;
    public int octaves = 4;
    public double persistence = 0.5;
    public double lacunarity = 2.0;
    public long seed = 42;
    public double flattenPower = 1.0;
    
    // Terrain categorization thresholds
    public double waterLevel = 0.3;
    public double hillLevel = 0.5;
    public double mountainLevel = 0.7;
    public double transition = 0.05;
}
