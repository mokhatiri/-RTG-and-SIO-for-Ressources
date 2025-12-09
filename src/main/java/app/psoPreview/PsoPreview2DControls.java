package app.psoPreview;

import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.VBox;
import javafx.scene.control.Accordion;
import javafx.scene.control.TitledPane;

public class PsoPreview2DControls {
    private PsoParameters psoParams;
    private final Runnable onUpdateDisplay;
    
    public PsoPreview2DControls(PsoParameters psoParams, Runnable onUpdateDisplay) {
        this.psoParams = psoParams;
        this.onUpdateDisplay = onUpdateDisplay;
    }

    public VBox buildControlsPanel() {
        VBox controls = new VBox(10);
        controls.setPadding(new javafx.geometry.Insets(10));
        controls.setPrefWidth(280);

        Accordion accordion = new Accordion();
        accordion.getPanes().add(buildPsoParametersPane());
        accordion.setExpandedPane(accordion.getPanes().get(0));

        controls.getChildren().add(accordion);
        return controls;
    }

    private TitledPane buildPsoParametersPane() {
        VBox psoParamsBox = new VBox(10);

        addSlider(psoParamsBox, "Weight (w)", 0.0, 1.0, psoParams.weight, 0.01,
                newVal -> { psoParams.weight = newVal; onUpdateDisplay.run(); });
        addSlider(psoParamsBox, "Cognitive Coefficient (c1)", 0.0, 3.0, psoParams.cognitiveCoeff, 0.1,
                newVal -> { psoParams.cognitiveCoeff = newVal; onUpdateDisplay.run(); });
        addSlider(psoParamsBox, "Social Coefficient (c2)", 0.0, 3.0, psoParams.socialCoeff, 0.1,
                newVal -> { psoParams.socialCoeff = newVal; onUpdateDisplay.run(); });
        addSlider(psoParamsBox, "Resource Coefficient", 0.0, 5.0, psoParams.resourceCoeff, 0.1,
                newVal -> { psoParams.resourceCoeff = newVal; onUpdateDisplay.run(); });
        addSlider(psoParamsBox, "Terrain Coefficient", 0.0, 5.0, psoParams.terrainCoeff, 0.1,
                newVal -> { psoParams.terrainCoeff = newVal; onUpdateDisplay.run(); });

        ScrollPane scrollPane = new ScrollPane(psoParamsBox);
        scrollPane.setFitToWidth(true);
        scrollPane.setPrefHeight(400);

        return new TitledPane("PSO Parameters", scrollPane);
    }

    private void addSlider(VBox container, String labelText, double min, double max, double initialValue, double step,
                           java.util.function.Consumer<Double> onValueChanged) {
        Label sliderLabel = new Label(labelText + ": " + String.format("%.2f", initialValue));
        Slider slider = new Slider(min, max, initialValue);
        slider.setBlockIncrement(step);
        slider.setMajorTickUnit((max - min) / 4);
        slider.setShowTickLabels(true);
        slider.setShowTickMarks(true);
        slider.valueProperty().addListener((obs, oldVal, newVal) -> {
            double val = Math.round(newVal.doubleValue() / step) * step;
            sliderLabel.setText(labelText + ": " + String.format("%.2f", val));
            onValueChanged.accept(val);
        });

        container.getChildren().addAll(sliderLabel, slider);
    }
}
