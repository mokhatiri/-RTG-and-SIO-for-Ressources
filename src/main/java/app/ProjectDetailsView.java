package app;

import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

public final class ProjectDetailsView {
    private ProjectDetailsView() {}

    public static void show() {
        Stage stage = new Stage();
        stage.setTitle("Project Details — RTG & PSO");
        stage.setScene(createScene(stage));
        stage.show();
    }

    private static Scene createScene(Stage stage) {
        Label title = new Label("Project Details");
        title.getStyleClass().add("title-label");

        Label subtitle = new Label("Random Terrain Generation + Particle Swarm Optimization (JavaFX visualization)");
        subtitle.getStyleClass().add("subtitle-label");

        VBox content = new VBox(12);
        content.setPadding(new Insets(14));
        content.getStyleClass().add("details-card");

        content.getChildren().addAll(
            sectionTitle("What this project is"),
            paragraph("A JavaFX visualization project that combines procedural terrain generation (noise-based) with a Particle Swarm Optimization (PSO) demo. It is meant to help you understand how parameter choices affect maps and how swarm search behaves on top of those maps."),

            sectionTitle("Core parts"),
            bullet("Terrain generation (RTG): builds a height map, categorizes terrain bands, computes a flatness map."),
            bullet("Resource generation: noise-vein style clustering with terrain/flatness biasing."),
            bullet("PSO visualization: particles search the grid; fitness blends resource desirability and terrain suitability."),

            sectionTitle("What you can learn / demonstrate"),
            bullet("How noise parameters change terrain structure."),
            bullet("How multi-layer scoring (terrain + resources) affects optimization."),
            bullet("How PSO parameters (w, c1, c2) influence exploration vs exploitation."),

            sectionTitle("Real scenarios"),
            bullet("Games: procedural maps + choosing spawn/base locations."),
            bullet("Site selection: multi-criteria decision (constraints + attractiveness layers)."),
            bullet("Robotics & swarms: intuition-building for distributed search behaviours."),

            sectionTitle("ND note"),
            paragraph("The repository also includes n-dimensional (ND) utilities so parts of the core logic can be extended beyond 2D later, while keeping the current 2D visual demos." )
        );

        ScrollPane scroll = new ScrollPane(content);
        scroll.getStyleClass().add("details-scroll");
        scroll.setFitToWidth(true);
        scroll.setFitToHeight(true);

        Button close = new Button("Close");
        close.getStyleClass().add("launcher-button");
        close.setOnAction(e -> stage.close());

        HBox footer = new HBox(8, close);
        footer.setPadding(new Insets(0, 0, 0, 0));

        VBox root = new VBox(12, title, subtitle, scroll, footer);
        root.setPadding(new Insets(18));

        Scene scene = new Scene(root, 780, 520);
        try {
            scene.getStylesheets().add(ProjectDetailsView.class.getResource("/styles.css").toExternalForm());
        } catch (Exception ex) {
            System.err.println("Warning: styles.css not found in resources — using default styling.");
        }
        return scene;
    }

    private static Label sectionTitle(String text) {
        Label l = new Label(text);
        l.getStyleClass().add("details-section-title");
        l.setWrapText(true);
        return l;
    }

    private static Label paragraph(String text) {
        Label l = new Label(text);
        l.getStyleClass().add("details-paragraph");
        l.setWrapText(true);
        return l;
    }

    private static Label bullet(String text) {
        Label l = new Label("• " + text);
        l.getStyleClass().add("details-bullet");
        l.setWrapText(true);
        return l;
    }
}
