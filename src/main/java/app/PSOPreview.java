package app;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane.TitlePaneLayout;

import app.noisePreview.MapGenerator;
import app.noisePreview.NoisePreview2DControls;
import app.noisePreview.ResourceGenerationParams;
import app.noisePreview.TerrainGenerationParams;
import app.noisePreview.TerrainRenderer;
import app.psoPreview.PsoParameters;
import app.psoPreview.PsoPreview2DControls;

import javafx.scene.Scene;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;

import javafx.scene.layout.VBox;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;

public class PSOPreview {

    private static final int DISPLAY_SIZE = 512;

    private final TerrainGenerationParams terrainParams = new TerrainGenerationParams();
    private final ResourceGenerationParams resourceParams = new ResourceGenerationParams();
    private final PsoParameters psoParams = new PsoParameters();

    public Scene showPSOPreview() {
        BorderPane root = new BorderPane();
        VBox Terraincontrols = new NoisePreview2DControls(terrainParams, resourceParams, () -> this.updateDisplay()).buildControlsPanel();
        VBox psoControls = new PsoPreview2DControls(psoParams, () -> this.updateDisplay()).buildControlsPanel();

        root.setLeft(Terraincontrols);
        root.setRight(psoControls);

        Scene scene = new Scene(root, DISPLAY_SIZE + 10 + Terraincontrols.getPrefWidth() + psoControls.getPrefWidth(), DISPLAY_SIZE + 20);

        return scene;
    }

    private void updateDisplay() {
        // nothing yet
    }
}
