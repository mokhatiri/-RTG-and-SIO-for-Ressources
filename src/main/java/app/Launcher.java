package app;

import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Separator;
import javafx.scene.effect.DropShadow;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.stage.Stage;

public class Launcher extends Application {

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("RTG & PSO Launcher");

        Label titleLabel = new Label("RTG & PSO Simulators");
        titleLabel.getStyleClass().add("title-label");
        titleLabel.setFont(Font.font("Segoe UI", FontWeight.BOLD, 28));
        titleLabel.setEffect(new DropShadow(6, Color.gray(0, 0.35)));

        Label subtitle = new Label("Explore procedurally generated terrain and swarm optimization in real-time");
        subtitle.getStyleClass().add("subtitle-label");

        Button noisePreviewButton = new Button("Noise & Resource Preview");
        noisePreviewButton.getStyleClass().add("launcher-button");
        noisePreviewButton.setPrefSize(220, 44);
        noisePreviewButton.setGraphic(new Label("🗺"));
        noisePreviewButton.setOnAction(e -> {
            Stage noiseStage = new Stage();
            NoisePreview2D noisePreview = new NoisePreview2D();
            Scene noiseScene = noisePreview.showNoisePreview2D();
            noiseStage.setScene(noiseScene);
            noiseStage.setTitle("Noise and Resource Preview 2D – Movable View");
            noiseStage.show();
        });

        Button psoPreviewButton = new Button("PSO Visualization");
        psoPreviewButton.getStyleClass().add("launcher-button");
        psoPreviewButton.setPrefSize(220, 44);
        psoPreviewButton.setGraphic(new Label("🐦"));
        psoPreviewButton.setOnAction(e -> {
            Stage psoStage = new Stage();
            PSOPreview psoPreview = new PSOPreview();
            Scene psoScene = psoPreview.showPSOPreview();
            psoStage.setScene(psoScene);
            psoStage.setTitle("Particle Swarm Optimization Visualization (JavaFX)");
            psoStage.show();
        });

        Button detailsButton = new Button("Project details");
        detailsButton.getStyleClass().add("launcher-button");
        detailsButton.setPrefSize(220, 44);
        detailsButton.setGraphic(new Label("ℹ"));
        detailsButton.setOnAction(e -> ProjectDetailsView.show());

        // subtle hover effects
        noisePreviewButton.setOnMouseEntered(e -> { noisePreviewButton.setScaleX(1.03); noisePreviewButton.setScaleY(1.03);} );
        noisePreviewButton.setOnMouseExited(e -> { noisePreviewButton.setScaleX(1.0); noisePreviewButton.setScaleY(1.0);} );
        psoPreviewButton.setOnMouseEntered(e -> { psoPreviewButton.setScaleX(1.03); psoPreviewButton.setScaleY(1.03);} );
        psoPreviewButton.setOnMouseExited(e -> { psoPreviewButton.setScaleX(1.0); psoPreviewButton.setScaleY(1.0);} );

        // Buttons container with spacing
        HBox buttonRow = new HBox(14, noisePreviewButton, psoPreviewButton);
        buttonRow.setAlignment(Pos.CENTER);

        HBox infoRow = new HBox(14, detailsButton);
        infoRow.setAlignment(Pos.CENTER);

        VBox root = new VBox(14); // Increased spacing
        root.setAlignment(Pos.CENTER);
        root.setPadding(new Insets(28)); // Added padding
        // footer with version and exit
        Label footerLabel = new Label("v0.1 — RTG & SIO");
        footerLabel.getStyleClass().add("footer-label");
        Button exitButton = new Button("Exit");
        exitButton.getStyleClass().add("launcher-button");
        exitButton.setOnAction(e -> primaryStage.close());

        HBox footer = new HBox(8);
        footer.setAlignment(Pos.CENTER_RIGHT);
        footer.getChildren().addAll(footerLabel, exitButton);
        root.getChildren().addAll(titleLabel, subtitle, new Separator(), buttonRow, infoRow, footer);

        Scene scene = new Scene(root, 560, 370); // Slightly taller for the extra button
        // apply application stylesheet
        try {
            scene.getStylesheets().add(getClass().getResource("/styles.css").toExternalForm());
        } catch (Exception ex) {
            System.err.println("Warning: styles.css not found in resources — using default styling.");
        }
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
