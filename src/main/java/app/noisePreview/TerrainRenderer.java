package app.noisePreview;

import terrain.NaturalResourceRandomizer;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

// Handles rendering of noise, terrain, and resource layers to canvases
public class TerrainRenderer {
    private static final int WIDTH = 128;
    private static final int HEIGHT = 128;
    private static final int DISPLAY_SIZE = 512;
    private static final int CELL_SIZE = DISPLAY_SIZE / WIDTH;

    private final GraphicsContext noiseGC;
    private final GraphicsContext terrainGC;

    public TerrainRenderer(GraphicsContext noiseGC, GraphicsContext terrainGC) {
        this.noiseGC = noiseGC;
        this.terrainGC = terrainGC;
    }

    public void renderAll(double[][] heightMap, int[][] terrainTypes, 
                         boolean[][][] resourceMaps, ResourceGenerationParams resourceParams) {
        renderNoise(heightMap);
        renderTerrain(terrainTypes);
        renderResources(resourceMaps, resourceParams);
    }

    private void renderNoise(double[][] heightMap) {
        for (int x = 0; x < WIDTH; x++) {
            for (int y = 0; y < HEIGHT; y++) {
                int gray = (int) (heightMap[x][y] * 255);
                gray = Math.max(0, Math.min(255, gray));
                noiseGC.setFill(Color.rgb(gray, gray, gray));
                noiseGC.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
            }
        }
    }

    private void renderTerrain(int[][] terrainTypes) {
        for (int x = 0; x < WIDTH; x++) {
            for (int y = 0; y < HEIGHT; y++) {
                Color terrainColor;
                switch (terrainTypes[x][y]) {
                    // Water progression: Deep -> Shallow -> Beach
                    case 0 -> terrainColor = Color.web("#1a4d7a");    // Deep water (dark blue)
                    case 1 -> terrainColor = Color.web("#2d7fa3");    // Shallow water (medium blue)
                    
                    // Coastal/Plains progression: Beach -> Grass -> Meadow
                    case 2 -> terrainColor = Color.web("#c2b280");    // Beach/sand (tan)
                    case 3 -> terrainColor = Color.web("#7cb342");    // Plains (light green)
                    
                    // Foothills/Hills progression: Grass -> Dark green -> Olive
                    case 4 -> terrainColor = Color.web("#558b2f");    // Foothills (medium-dark green)
                    case 5 -> terrainColor = Color.web("#33691e");    // Hills (dark green)
                    
                    // Mountains progression: Rocky -> Grey -> Peak
                    case 6 -> terrainColor = Color.web("#8d7662");    // Rocky transition (taupe)
                    case 7 -> terrainColor = Color.web("#4a4a4a");    // Mountains (dark grey)
                    
                    default -> terrainColor = Color.BLACK;
                }
                terrainGC.setFill(terrainColor);
                terrainGC.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
            }
        }
    }

    private void renderResources(boolean[][][] resourceMaps, ResourceGenerationParams resourceParams) {
        for (int x = 0; x < WIDTH; x++) {
            for (int y = 0; y < HEIGHT; y++) {
                if (resourceParams.showSedimentary && resourceMaps[x][y][NaturalResourceRandomizer.SEDIMENTARY_ROCK]) {
                    terrainGC.setFill(resourceParams.sedimentaryColor);
                    terrainGC.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                }
                if (resourceParams.showGemstones && resourceMaps[x][y][NaturalResourceRandomizer.GEMSTONES]) {
                    terrainGC.setFill(resourceParams.gemstonesColor);
                    terrainGC.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                }
                if (resourceParams.showIronOre && resourceMaps[x][y][NaturalResourceRandomizer.IRON_ORE]) {
                    terrainGC.setFill(resourceParams.ironOreColor);
                    terrainGC.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                }
                if (resourceParams.showCoal && resourceMaps[x][y][NaturalResourceRandomizer.COAL]) {
                    terrainGC.setFill(resourceParams.coalColor);
                    terrainGC.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                }
                if (resourceParams.showGoldOre && resourceMaps[x][y][NaturalResourceRandomizer.GOLD_ORE]) {
                    terrainGC.setFill(resourceParams.goldOreColor);
                    terrainGC.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                }
                if (resourceParams.showWood && resourceMaps[x][y][NaturalResourceRandomizer.WOOD]) {
                    terrainGC.setFill(resourceParams.woodColor);
                    terrainGC.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                }
                if (resourceParams.showCattleHerd && resourceMaps[x][y][NaturalResourceRandomizer.CATTLE_HERD]) {
                    terrainGC.setFill(resourceParams.cattleHerdColor);
                    terrainGC.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                }
                if (resourceParams.showWolfPack && resourceMaps[x][y][NaturalResourceRandomizer.WOLF_PACK]) {
                    terrainGC.setFill(resourceParams.wolfPackColor);
                    terrainGC.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                }
                if (resourceParams.showFishSchool && resourceMaps[x][y][NaturalResourceRandomizer.FISH_SCHOOL]) {
                    terrainGC.setFill(resourceParams.fishSchoolColor);
                    terrainGC.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                }
            }
        }
    }
}
