package app.noisePreview;

import terrain.NaturalResourceRandomizer;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

// Handles rendering of noise, terrain, and resource layers to canvases
public class TerrainRenderer {
    private static final int WIDTH = 128;
    private static final int HEIGHT = 128;

    private final GraphicsContext noiseGC;
    private final GraphicsContext terrainGC;

    public TerrainRenderer(GraphicsContext noiseGC, GraphicsContext terrainGC) {
        this.noiseGC = noiseGC;
        this.terrainGC = terrainGC;
    }

    public void renderAll(double[][] heightMap, int[][] terrainTypes, 
                         boolean[][][] resourceMaps, ResourceGenerationParams resourceParams) {
        if (noiseGC != null) {
            renderNoise(heightMap);
        }
        renderTerrain(terrainTypes);
        renderResources(resourceMaps, resourceParams);
    }

    private void renderNoise(double[][] heightMap) {
        clear(noiseGC);
        CellLayout layout = computeLayout(noiseGC);
        for (int x = 0; x < WIDTH; x++) {
            for (int y = 0; y < HEIGHT; y++) {
                int gray = (int) (heightMap[x][y] * 255);
                gray = Math.max(0, Math.min(255, gray));
                noiseGC.setFill(Color.rgb(gray, gray, gray));
                noiseGC.fillRect(
                        layout.originX + x * layout.cellSize,
                        layout.originY + y * layout.cellSize,
                        layout.cellSize,
                        layout.cellSize
                );
            }
        }
    }

    private void renderTerrain(int[][] terrainTypes) {
        clear(terrainGC);
        CellLayout layout = computeLayout(terrainGC);
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
                terrainGC.fillRect(
                        layout.originX + x * layout.cellSize,
                        layout.originY + y * layout.cellSize,
                        layout.cellSize,
                        layout.cellSize
                );
            }
        }
    }

    private void renderResources(boolean[][][] resourceMaps, ResourceGenerationParams resourceParams) {
        CellLayout layout = computeLayout(terrainGC);
        for (int x = 0; x < WIDTH; x++) {
            for (int y = 0; y < HEIGHT; y++) {
                if (resourceParams.showSedimentary && resourceMaps[x][y][NaturalResourceRandomizer.SEDIMENTARY_ROCK]) {
                    terrainGC.setFill(resourceParams.sedimentaryColor);
                    terrainGC.fillRect(layout.originX + x * layout.cellSize, layout.originY + y * layout.cellSize, layout.cellSize, layout.cellSize);
                }
                if (resourceParams.showGemstones && resourceMaps[x][y][NaturalResourceRandomizer.GEMSTONES]) {
                    terrainGC.setFill(resourceParams.gemstonesColor);
                    terrainGC.fillRect(layout.originX + x * layout.cellSize, layout.originY + y * layout.cellSize, layout.cellSize, layout.cellSize);
                }
                if (resourceParams.showIronOre && resourceMaps[x][y][NaturalResourceRandomizer.IRON_ORE]) {
                    terrainGC.setFill(resourceParams.ironOreColor);
                    terrainGC.fillRect(layout.originX + x * layout.cellSize, layout.originY + y * layout.cellSize, layout.cellSize, layout.cellSize);
                }
                if (resourceParams.showCoal && resourceMaps[x][y][NaturalResourceRandomizer.COAL]) {
                    terrainGC.setFill(resourceParams.coalColor);
                    terrainGC.fillRect(layout.originX + x * layout.cellSize, layout.originY + y * layout.cellSize, layout.cellSize, layout.cellSize);
                }
                if (resourceParams.showGoldOre && resourceMaps[x][y][NaturalResourceRandomizer.GOLD_ORE]) {
                    terrainGC.setFill(resourceParams.goldOreColor);
                    terrainGC.fillRect(layout.originX + x * layout.cellSize, layout.originY + y * layout.cellSize, layout.cellSize, layout.cellSize);
                }
                if (resourceParams.showWood && resourceMaps[x][y][NaturalResourceRandomizer.WOOD]) {
                    terrainGC.setFill(resourceParams.woodColor);
                    terrainGC.fillRect(layout.originX + x * layout.cellSize, layout.originY + y * layout.cellSize, layout.cellSize, layout.cellSize);
                }
                if (resourceParams.showCattleHerd && resourceMaps[x][y][NaturalResourceRandomizer.CATTLE_HERD]) {
                    terrainGC.setFill(resourceParams.cattleHerdColor);
                    terrainGC.fillRect(layout.originX + x * layout.cellSize, layout.originY + y * layout.cellSize, layout.cellSize, layout.cellSize);
                }
                if (resourceParams.showWolfPack && resourceMaps[x][y][NaturalResourceRandomizer.WOLF_PACK]) {
                    terrainGC.setFill(resourceParams.wolfPackColor);
                    terrainGC.fillRect(layout.originX + x * layout.cellSize, layout.originY + y * layout.cellSize, layout.cellSize, layout.cellSize);
                }
                if (resourceParams.showFishSchool && resourceMaps[x][y][NaturalResourceRandomizer.FISH_SCHOOL]) {
                    terrainGC.setFill(resourceParams.fishSchoolColor);
                    terrainGC.fillRect(layout.originX + x * layout.cellSize, layout.originY + y * layout.cellSize, layout.cellSize, layout.cellSize);
                }
            }
        }
    }

    private void clear(GraphicsContext gc) {
        if (gc == null || gc.getCanvas() == null) {
            return;
        }
        gc.clearRect(0, 0, gc.getCanvas().getWidth(), gc.getCanvas().getHeight());
    }

    private CellLayout computeLayout(GraphicsContext gc) {
        double w = (gc == null || gc.getCanvas() == null) ? 0 : gc.getCanvas().getWidth();
        double h = (gc == null || gc.getCanvas() == null) ? 0 : gc.getCanvas().getHeight();
        if (w <= 0 || h <= 0) {
            return new CellLayout(0, 0, 0);
        }
        double cellSize = Math.min(w / WIDTH, h / HEIGHT);
        double drawW = cellSize * WIDTH;
        double drawH = cellSize * HEIGHT;
        double originX = (w - drawW) / 2.0;
        double originY = (h - drawH) / 2.0;
        return new CellLayout(originX, originY, cellSize);
    }

    private static final class CellLayout {
        private final double originX;
        private final double originY;
        private final double cellSize;

        private CellLayout(double originX, double originY, double cellSize) {
            this.originX = originX;
            this.originY = originY;
            this.cellSize = cellSize;
        }
    }
}
