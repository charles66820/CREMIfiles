package com.cgoedefroit.tp3app;

import com.cgoedefroit.tp3.AxesAlignedRectangle;
import com.cgoedefroit.tp3.Point2D;
import com.cgoedefroit.tp3.Triangle;
import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class tp3app extends Application {

    public void start(Stage stage) throws Exception {

        AxesAlignedRectangle rectangle = new AxesAlignedRectangle(new Point2D(40, 60), 50, 150);
        rectangle.setR(255);

        com.cgoedefroit.tp3.Circle circle = new com.cgoedefroit.tp3.Circle(new Point2D(20, 40), 10);
        circle.setG(255);

        Triangle triangle = new Triangle(new Point2D(50, 50), new Point2D(10, 80), new Point2D(60, 80));
        triangle.setB(255);

        Group root = new Group();
        root.getChildren().add(rectangle.toShapeFX());
        root.getChildren().add(circle.toShapeFX());
        root.getChildren().add(triangle.toShapeFX());

        Scene scene = new Scene(root, 400, 300);

        stage.setScene(scene);
        stage.show();
    }

    public static void main(String... args) {
        Application.launch(args);
    }
}