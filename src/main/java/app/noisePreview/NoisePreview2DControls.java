package app.noisePreview;

import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.control.Accordion;
import javafx.scene.control.TitledPane;

// Builds UI controls for terrain and resource generation parameters
public class NoisePreview2DControls {
    private final TerrainGenerationParams terrainParams;
    private final ResourceGenerationParams resourceParams;
    private final Runnable onUpdateDisplay;

    public NoisePreview2DControls(TerrainGenerationParams terrainParams, 
                                  ResourceGenerationParams resourceParams,
                                  Runnable onUpdateDisplay) {
        this.terrainParams = terrainParams;
        this.resourceParams = resourceParams;
        this.onUpdateDisplay = onUpdateDisplay;
    }

    public VBox buildControlsPanel() {
        VBox controls = new VBox(10);
        controls.setPadding(new javafx.geometry.Insets(10));
        controls.setPrefWidth(280);

        Accordion accordion = new Accordion();
        accordion.getPanes().add(buildNoiseGenerationPane());
        accordion.getPanes().add(buildResourceGenerationPane());
        accordion.setExpandedPane(accordion.getPanes().get(0));

        controls.getChildren().add(accordion);
        return controls;
    }

    private TitledPane buildNoiseGenerationPane() {
        VBox noiseControlsVBox = new VBox(5);

        addIntSlider(noiseControlsVBox, "Octaves.", 1, 30, terrainParams.octaves, 1,
                newVal -> terrainParams.octaves = newVal);
        addTerrainSlider(noiseControlsVBox, "Persistence.", 0.1, 1.0, terrainParams.persistence, 0.05,
                newVal -> terrainParams.persistence = newVal);
        addTerrainSlider(noiseControlsVBox, "Lacunarity.", 1.0, 4.0, terrainParams.lacunarity, 0.1,
                newVal -> terrainParams.lacunarity = newVal);
        addLongSlider(noiseControlsVBox, "Seed.", 0, 10000, terrainParams.seed, 1,
                newVal -> terrainParams.seed = newVal);
        addTerrainSlider(noiseControlsVBox, "Flatten Power.", 0.1, 5.0, terrainParams.flattenPower, 0.1,
                newVal -> terrainParams.flattenPower = newVal);
        addTerrainSlider(noiseControlsVBox, "Water Level.", 0.0, 1.0, terrainParams.waterLevel, 0.05,
                newVal -> terrainParams.waterLevel = newVal);
        addTerrainSlider(noiseControlsVBox, "Hill Level.", 0.0, 1.0, terrainParams.hillLevel, 0.05,
                newVal -> terrainParams.hillLevel = newVal);
        addTerrainSlider(noiseControlsVBox, "Mountain Level.", 0.0, 1.0, terrainParams.mountainLevel, 0.05,
                newVal -> terrainParams.mountainLevel = newVal);
        addTerrainSlider(noiseControlsVBox, "Transition Width.", 0.0, 0.2, terrainParams.transition, 0.01,
                newVal -> terrainParams.transition = newVal);

        ScrollPane noiseScrollPane = new ScrollPane(noiseControlsVBox);
        noiseScrollPane.setFitToWidth(true);
        noiseScrollPane.setPrefHeight(400);

        return new TitledPane("Noise Generation", noiseScrollPane);
    }

    private TitledPane buildResourceGenerationPane() {
        VBox resourceControlsVBox = new VBox(5);

        addResourceControl(resourceControlsVBox, "Sedimentary Rock Prob.",
                0.0, 1.0, resourceParams.sedimentaryRockProb, 0.01,
                newVal -> resourceParams.sedimentaryRockProb = newVal,
                () -> resourceParams.showSedimentary, newVal -> resourceParams.showSedimentary = newVal,
                () -> resourceParams.sedimentaryColor, newVal -> resourceParams.sedimentaryColor = newVal);

        addResourceControl(resourceControlsVBox, "Gemstones Prob.",
                0.0, 1.0, resourceParams.gemstonesProb, 0.01,
                newVal -> resourceParams.gemstonesProb = newVal,
                () -> resourceParams.showGemstones, newVal -> resourceParams.showGemstones = newVal,
                () -> resourceParams.gemstonesColor, newVal -> resourceParams.gemstonesColor = newVal);

        addResourceControl(resourceControlsVBox, "Iron Ore Prob.",
                0.0, 1.0, resourceParams.ironOreProb, 0.01,
                newVal -> resourceParams.ironOreProb = newVal,
                () -> resourceParams.showIronOre, newVal -> resourceParams.showIronOre = newVal,
                () -> resourceParams.ironOreColor, newVal -> resourceParams.ironOreColor = newVal);

        addResourceControl(resourceControlsVBox, "Coal Prob.",
                0.0, 1.0, resourceParams.coalProb, 0.01,
                newVal -> resourceParams.coalProb = newVal,
                () -> resourceParams.showCoal, newVal -> resourceParams.showCoal = newVal,
                () -> resourceParams.coalColor, newVal -> resourceParams.coalColor = newVal);

        addResourceControl(resourceControlsVBox, "Gold Ore Prob.",
                0.0, 1.0, resourceParams.goldOreProb, 0.01,
                newVal -> resourceParams.goldOreProb = newVal,
                () -> resourceParams.showGoldOre, newVal -> resourceParams.showGoldOre = newVal,
                () -> resourceParams.goldOreColor, newVal -> resourceParams.goldOreColor = newVal);

        addResourceControl(resourceControlsVBox, "Wood Prob.",
                0.0, 1.0, resourceParams.woodProb, 0.01,
                newVal -> resourceParams.woodProb = newVal,
                () -> resourceParams.showWood, newVal -> resourceParams.showWood = newVal,
                () -> resourceParams.woodColor, newVal -> resourceParams.woodColor = newVal);

        addResourceControl(resourceControlsVBox, "Cattle Herd Prob.",
                0.0, 1.0, resourceParams.cattleHerdProb, 0.01,
                newVal -> resourceParams.cattleHerdProb = newVal,
                () -> resourceParams.showCattleHerd, newVal -> resourceParams.showCattleHerd = newVal,
                () -> resourceParams.cattleHerdColor, newVal -> resourceParams.cattleHerdColor = newVal);

        addResourceControl(resourceControlsVBox, "Wolf Pack Prob.",
                0.0, 1.0, resourceParams.wolfPackProb, 0.01,
                newVal -> resourceParams.wolfPackProb = newVal,
                () -> resourceParams.showWolfPack, newVal -> resourceParams.showWolfPack = newVal,
                () -> resourceParams.wolfPackColor, newVal -> resourceParams.wolfPackColor = newVal);

        addResourceControl(resourceControlsVBox, "Fish School Prob.",
                0.0, 1.0, resourceParams.fishSchoolProb, 0.01,
                newVal -> resourceParams.fishSchoolProb = newVal,
                () -> resourceParams.showFishSchool, newVal -> resourceParams.showFishSchool = newVal,
                () -> resourceParams.fishSchoolColor, newVal -> resourceParams.fishSchoolColor = newVal);

        addLongSlider(resourceControlsVBox, "Resource Randomizer Seed.", 0, 10000, resourceParams.randomizerSeed, 1,
                newVal -> resourceParams.randomizerSeed = newVal);

        ScrollPane resourceScrollPane = new ScrollPane(resourceControlsVBox);
        resourceScrollPane.setFitToWidth(true);
        resourceScrollPane.setPrefHeight(400);

        return new TitledPane("Resource Generation", resourceScrollPane);
    }

    private void addTerrainSlider(VBox parent, String label, double min, double max, 
                                   double initial, double blockIncrement, Consumer<Double> onUpdate) {
        Label sliderLabel = new Label(label);
        Slider slider = createSlider(min, max, initial, blockIncrement);
        slider.valueProperty().addListener((obs, oldVal, newVal) -> {
            onUpdate.accept(newVal.doubleValue());
            onUpdateDisplay.run();
        });
        parent.getChildren().addAll(sliderLabel, slider);
    }

    private void addIntSlider(VBox parent, String label, double min, double max,
                             int initial, double blockIncrement, Consumer<Integer> onUpdate) {
        Label sliderLabel = new Label(label);
        Slider slider = createSlider(min, max, initial, blockIncrement);
        slider.valueProperty().addListener((obs, oldVal, newVal) -> {
            onUpdate.accept((int) newVal.doubleValue());
            onUpdateDisplay.run();
        });
        parent.getChildren().addAll(sliderLabel, slider);
    }

    private void addLongSlider(VBox parent, String label, double min, double max,
                              long initial, double blockIncrement, Consumer<Long> onUpdate) {
        Label sliderLabel = new Label(label);
        Slider slider = createSlider(min, max, initial, blockIncrement);
        slider.valueProperty().addListener((obs, oldVal, newVal) -> {
            onUpdate.accept((long) newVal.doubleValue());
            onUpdateDisplay.run();
        });
        parent.getChildren().addAll(sliderLabel, slider);
    }

    private void addResourceControl(VBox parent, String label, double min, double max, 
                                     double initial, double blockIncrement, Consumer<Double> onProbUpdate,
                                     BooleanGetter showGetter, BooleanSetter showSetter,
                                     ColorGetter colorGetter, ColorSetter colorSetter) {
        Label probLabel = new Label(label);
        Slider probSlider = createSlider(min, max, initial, blockIncrement);
        probSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            onProbUpdate.accept(newVal.doubleValue());
            onUpdateDisplay.run();
        });

        CheckBox showCheckBox = new CheckBox("Show");
        showCheckBox.setSelected(showGetter.get());
        showCheckBox.selectedProperty().addListener((obs, oldVal, newVal) -> {
            showSetter.set(newVal);
            onUpdateDisplay.run();
        });

        javafx.scene.paint.Color initialColor = colorGetter.get();
        javafx.scene.control.ColorPicker colorPicker = new javafx.scene.control.ColorPicker(initialColor);
        colorPicker.valueProperty().addListener((obs, oldVal, newVal) -> {
            colorSetter.set(newVal);
            onUpdateDisplay.run();
        });

        HBox controlBox = new HBox(5, showCheckBox, colorPicker);
        parent.getChildren().addAll(probLabel, probSlider, controlBox);
    }

    private Slider createSlider(double min, double max, double initial, double blockIncrement) {
        Slider slider = new Slider(min, max, initial);
        slider.setBlockIncrement(blockIncrement);
        slider.setShowTickLabels(true);
        slider.setShowTickMarks(true);
        slider.setMajorTickUnit((max - min) / 4);
        slider.setMinorTickCount(5);
        return slider;
    }

    // Functional interfaces for type-safe callbacks
    @FunctionalInterface
    interface Consumer<T> {
        void accept(T value);
    }

    @FunctionalInterface
    interface BooleanGetter {
        boolean get();
    }

    @FunctionalInterface
    interface BooleanSetter {
        void set(boolean value);
    }

    @FunctionalInterface
    interface ColorGetter {
        javafx.scene.paint.Color get();
    }

    @FunctionalInterface
    interface ColorSetter {
        void set(javafx.scene.paint.Color value);
    }
}
