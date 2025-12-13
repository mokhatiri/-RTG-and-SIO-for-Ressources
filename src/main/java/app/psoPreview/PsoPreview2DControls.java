package app.psoPreview;

import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Button;
import javafx.scene.control.ColorPicker;
import javafx.scene.layout.VBox;
import javafx.scene.control.Accordion;
import javafx.scene.control.TitledPane;
import javafx.scene.layout.HBox;

public class PsoPreview2DControls {
    private PsoParameters psoParams;
        private final Runnable onApplyChanges;
        private final Runnable onColorsChanged;
    private final Runnable onToggleRun;
    private final Runnable onStep;
    private final Runnable onReset;

        // Draft values: user can edit sliders without immediately affecting simulation
        private double draftWeight;
        private double draftCognitive;
        private double draftSocial;
        private double draftResourceCoeff;
        private double draftTerrainCoeff;
        private int draftParticleCount;
        private int draftIterationsPerTick;
    
        public PsoPreview2DControls(PsoParameters psoParams,
                                                                Runnable onApplyChanges,
                                                                Runnable onColorsChanged,
                                                                Runnable onToggleRun,
                                                                Runnable onStep,
                                                                Runnable onReset) {
        this.psoParams = psoParams;
                this.onApplyChanges = onApplyChanges;
                this.onColorsChanged = onColorsChanged;
        this.onToggleRun = onToggleRun;
        this.onStep = onStep;
        this.onReset = onReset;

                this.draftWeight = psoParams.weight;
                this.draftCognitive = psoParams.cognitiveCoeff;
                this.draftSocial = psoParams.socialCoeff;
                this.draftResourceCoeff = psoParams.resourceCoeff;
                this.draftTerrainCoeff = psoParams.terrainCoeff;
                this.draftParticleCount = psoParams.particleCount;
                this.draftIterationsPerTick = psoParams.iterationsPerTick;
    }

    public VBox buildControlsPanel() {
        VBox controls = new VBox(10);
        controls.setPadding(new javafx.geometry.Insets(10));
        controls.setPrefWidth(280);

        Accordion accordion = new Accordion();
        accordion.getPanes().add(buildRunPane());
        accordion.getPanes().add(buildPsoParametersPane());
        accordion.setExpandedPane(accordion.getPanes().get(0));

        Button apply = new Button("Apply changes");
        apply.setOnAction(e -> {
            // Copy drafts into live parameters
            psoParams.weight = draftWeight;
            psoParams.cognitiveCoeff = draftCognitive;
            psoParams.socialCoeff = draftSocial;
            psoParams.resourceCoeff = draftResourceCoeff;
            psoParams.terrainCoeff = draftTerrainCoeff;
            psoParams.particleCount = draftParticleCount;
            psoParams.iterationsPerTick = draftIterationsPerTick;
            onApplyChanges.run();
        });


        controls.getChildren().add(accordion);        
        controls.getChildren().add(apply);

        return controls;
    }

    private TitledPane buildRunPane() {
        VBox box = new VBox(10);

        HBox row = new HBox(8);
        Button toggle = new Button("Start / Pause");
        Button step = new Button("Step");
        Button reset = new Button("Reset");

        toggle.setOnAction(e -> onToggleRun.run());
        step.setOnAction(e -> onStep.run());
        reset.setOnAction(e -> onReset.run());

        row.getChildren().addAll(toggle, step, reset);

        addIntSlider(box, "Particles", 5, 250, draftParticleCount, 1,
                newVal -> draftParticleCount = newVal);
        addIntSlider(box, "Iterations / tick", 1, 50, draftIterationsPerTick, 1,
                newVal -> draftIterationsPerTick = newVal);

        box.getChildren().add(buildColorPickersRow());

        box.getChildren().add(0, row);
        return new TitledPane("Run", box);
    }

    private VBox buildColorPickersRow() {
        VBox colorBox = new VBox(8);

        colorBox.getChildren().add(buildColorPicker("Particle color", psoParams.particleColor,
                c -> { psoParams.particleColor = c; onColorsChanged.run(); }));

        colorBox.getChildren().add(buildColorPicker("Personal best color", psoParams.personalBestColor,
                c -> { psoParams.personalBestColor = c; onColorsChanged.run(); }));

        colorBox.getChildren().add(buildColorPicker("Global best color", psoParams.globalBestColor,
                c -> { psoParams.globalBestColor = c; onColorsChanged.run(); }));

        return colorBox;
    }

    private HBox buildColorPicker(String label, javafx.scene.paint.Color initial,
                                 java.util.function.Consumer<javafx.scene.paint.Color> onChanged) {
        Label l = new Label(label);
        ColorPicker picker = new ColorPicker(initial);
        picker.valueProperty().addListener((obs, oldVal, newVal) -> onChanged.accept(newVal));
        HBox row = new HBox(8, l, picker);
        return row;
    }

    private TitledPane buildPsoParametersPane() {
        VBox psoParamsBox = new VBox(10);

        addSlider(psoParamsBox, "Weight (w)", 0.0, 1.0, draftWeight, 0.01,
                newVal -> draftWeight = newVal);
        addSlider(psoParamsBox, "Cognitive Coefficient (c1)", 0.0, 3.0, draftCognitive, 0.1,
                newVal -> draftCognitive = newVal);
        addSlider(psoParamsBox, "Social Coefficient (c2)", 0.0, 3.0, draftSocial, 0.1,
                newVal -> draftSocial = newVal);
        addSlider(psoParamsBox, "Resource Coefficient", 0.0, 5.0, draftResourceCoeff, 0.1,
                newVal -> draftResourceCoeff = newVal);
        addSlider(psoParamsBox, "Terrain Coefficient", 0.0, 5.0, draftTerrainCoeff, 0.1,
                newVal -> draftTerrainCoeff = newVal);

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

    private void addIntSlider(VBox container, String labelText, int min, int max, int initialValue, int step,
                              java.util.function.Consumer<Integer> onValueChanged) {
        Label sliderLabel = new Label(labelText + ": " + initialValue);
        Slider slider = new Slider(min, max, initialValue);
        slider.setBlockIncrement(step);
        slider.setMajorTickUnit(Math.max(1, (max - min) / 4.0));
        slider.setShowTickLabels(true);
        slider.setShowTickMarks(true);
        slider.valueProperty().addListener((obs, oldVal, newVal) -> {
            int val = (int) Math.round(newVal.doubleValue() / step) * step;
            if (val < min) val = min;
            if (val > max) val = max;
            sliderLabel.setText(labelText + ": " + val);
            onValueChanged.accept(val);
        });

        container.getChildren().addAll(sliderLabel, slider);
    }
}
