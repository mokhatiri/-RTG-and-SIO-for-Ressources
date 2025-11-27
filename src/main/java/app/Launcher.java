package app;

import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.stage.Stage;

public class Launcher extends Application {

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("RTG and PSO Launcher");

        Label titleLabel = new Label("Welcome to the Simulators!");
        titleLabel.setFont(Font.font("Arial", FontWeight.BOLD, 24));

        Button noisePreviewButton = new Button("Noise & Resource Preview");
        noisePreviewButton.setPrefSize(200, 40);
        noisePreviewButton.setOnAction(e -> {
            Stage noiseStage = new Stage();
            NoisePreview2D noisePreview = new NoisePreview2D();
            Scene noiseScene = noisePreview.showNoisePreview2D();
            noiseStage.setScene(noiseScene);
            noiseStage.setTitle("Noise and Resource Preview 2D – Movable View");
            noiseStage.show();
        });

        Button psoPreviewButton = new Button("PSO Visualization");
        psoPreviewButton.setPrefSize(200, 40);
        psoPreviewButton.setOnAction(e -> {
            Stage psoStage = new Stage();
            PSOPreview psoPreview = new PSOPreview();
            Scene psoScene = psoPreview.showPSOPreview();
            psoStage.setScene(psoScene);
            psoStage.setTitle("Particle Swarm Optimization Visualization (JavaFX)");
            psoStage.show();
        });

        VBox root = new VBox(20); // Increased spacing
        root.setAlignment(Pos.CENTER);
        root.setPadding(new Insets(30)); // Added padding
        root.getChildren().addAll(titleLabel, noisePreviewButton, psoPreviewButton);

        Scene scene = new Scene(root, 400, 300); // Increased window size
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
