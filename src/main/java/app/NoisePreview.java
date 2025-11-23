package app;

import terrain.NoiseMapGenerator;

import javafx.application.Application;

import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;

import javafx.scene.control.Label;
import javafx.scene.control.Slider;

import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;

import javafx.scene.layout.VBox;
import javafx.stage.Stage;

public class NoisePreview extends Application { // javaFx basic format

    private static final int WIDTH = 512; // width of the canvas
    private static final int HEIGHT = 512; // height of the canvas

    @Override
    public void start(Stage stage) {
        VBox root = new VBox(10); // vertical layout with spacing

        // Canvas to draw the noise map
        Canvas canvas = new Canvas(WIDTH, HEIGHT);
        GraphicsContext gc = canvas.getGraphicsContext2D();

        // Sliders
        Slider scaleSlider = new Slider(1, 500, 100);
        Slider octavesSlider = new Slider(1, 8, 4);
        Slider persistenceSlider = new Slider(0.1, 1.0, 0.5);
        Slider lacunaritySlider = new Slider(1.0, 4.0, 2.0);
        Slider seedSlider = new Slider(0, 10000, 42);

        root.getChildren().addAll(
                new Label("Scale"), scaleSlider,
                new Label("Octaves"), octavesSlider,
                new Label("Persistence"), persistenceSlider,
                new Label("Lacunarity"), lacunaritySlider,
                new Label("Seed"), seedSlider,
                canvas
        );

        Runnable drawNoise = () -> {
            int octaves = (int) octavesSlider.getValue();
            double scale = scaleSlider.getValue();
            double persistence = persistenceSlider.getValue();
            double lacunarity = lacunaritySlider.getValue();
            long seed = (long) seedSlider.getValue();

            NoiseMapGenerator generator = new NoiseMapGenerator(seed); // create generator with seed
            double[][] map = generator.generate(WIDTH, HEIGHT, scale, octaves, persistence, lacunarity);

            WritableImage image = new WritableImage(WIDTH, HEIGHT); // create image to draw on
            PixelWriter pw = image.getPixelWriter(); // get pixel writer to set pixels

            // Convert noise values to grayscale
            for (int x = 0; x < WIDTH; x++) {
                for (int y = 0; y < HEIGHT; y++) {
                    double v = map[x][y];
                    int gray = (int) (v * 255);
                    gray = Math.min(255, Math.max(0, gray));
                    pw.setArgb(x, y, 0xFF000000 | (gray << 16) | (gray << 8) | gray);
                }
            }

            gc.drawImage(image, 0, 0);
        };

        // Redraw when sliders change
        // using listeners to detect changes
        scaleSlider.valueProperty().addListener(e -> drawNoise.run());
        octavesSlider.valueProperty().addListener(e -> drawNoise.run());
        persistenceSlider.valueProperty().addListener(e -> drawNoise.run());
        lacunaritySlider.valueProperty().addListener(e -> drawNoise.run());
        seedSlider.valueProperty().addListener(e -> drawNoise.run());

        drawNoise.run(); // initial draw

        Scene scene = new Scene(root); // create scene everything inside the window : root

        stage.setTitle("Noise Map Preview"); // title of the window
        stage.setScene(scene); // set the scene to the stage
        stage.show(); // show the stage
    }

    public static void main(String[] args) {
        launch(); // lanch the app
    }
}
