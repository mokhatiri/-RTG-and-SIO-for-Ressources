package app.noisePreview;

import javafx.scene.paint.Color;

// Encapsulates resource generation parameters, probabilities, visibility, and colors
public class ResourceGenerationParams {
    // Noise scale for resource distribution
    public double scale = 2.0;

    // Resource probabilities
    public double sedimentaryRockProb = 0.25;
    public double gemstonesProb = 0.25;
    public double ironOreProb = 0.25;
    public double coalProb = 0.25;
    public double goldOreProb = 0.25;
    public double woodProb = 0.25;
    public double cattleHerdProb = 0.25;
    public double wolfPackProb = 0.25;
    public double fishSchoolProb = 0.25;
    public long randomizerSeed = 12345;

    // Resource visibility toggles
    public boolean showSedimentary = false;
    public boolean showGemstones = false;
    public boolean showIronOre = false;
    public boolean showCoal = false;
    public boolean showGoldOre = false;
    public boolean showWood = false;
    public boolean showCattleHerd = false;
    public boolean showWolfPack = false;
    public boolean showFishSchool = false;

    // Resource colors
    public Color sedimentaryColor = Color.web("#D2B48C", 0.7);
    public Color gemstonesColor = Color.web("#FF00FF", 0.7);
    public Color ironOreColor = Color.web("#8B0000", 0.7);
    public Color coalColor = Color.web("#000000", 0.7);
    public Color goldOreColor = Color.web("#FFD700", 0.7);
    public Color woodColor = Color.web("#228B22", 0.7);
    public Color cattleHerdColor = Color.web("#D2691E", 0.7);
    public Color wolfPackColor = Color.web("#4B0082", 0.7);
    public Color fishSchoolColor = Color.web("#00CED1", 0.7);

    // Helper method to get all probabilities as an array
    public double[] getProbabilitiesArray() {
        return new double[]{
                sedimentaryRockProb,
                gemstonesProb,
                ironOreProb,
                coalProb,
                goldOreProb,
                woodProb,
                cattleHerdProb,
                wolfPackProb,
                fishSchoolProb
        };
    }
}
