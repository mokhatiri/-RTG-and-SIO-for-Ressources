package app.psoPreview;

import javafx.scene.paint.Color;

public  class PsoParameters {
    
    // double w, double c1, double c2, double resource_coeff, double terrain_coeff
    public double weight = 0.5;
    public double cognitiveCoeff = 1.5;
    public double socialCoeff = 1.5;
    public double resourceCoeff = 1.0;
    public double terrainCoeff = 1.0;

    public int particleCount = 60;
    public int iterationsPerTick = 2;

    // Visualization colors
    public Color particleColor = Color.rgb(255, 80, 80, 0.85);
    public Color personalBestColor = Color.rgb(80, 160, 255, 0.65);
    public Color globalBestColor = Color.rgb(255, 255, 255, 0.90);

}
