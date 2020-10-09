package com.cgoedefroit.tp3app;

import com.cgoedefroit.tp3.shape.Shape2D;
import com.cgoedefroit.tp3.util.tools;
import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.util.ArrayList;

public class tp3app extends Application {

    public void start(Stage stage) throws Exception {

        Group root = new Group();
        ArrayList<Shape2D> sList = tools.randShape2DList(10, 10, 200, 100, 400, 200);

        for (Shape2D oneShape : sList) root.getChildren().add(oneShape.toShapeFX());

        Scene scene = new Scene(root, 800, 450);

        stage.setScene(scene);
        stage.setTitle("Rand Shape2D List");
        stage.show();

        Stage secondStage = new Stage();
        Group root2 = new Group();
        ArrayList<Shape2D> sList2 = tools.circleInPolygoneList();

        for (Shape2D oneShape : sList2) root2.getChildren().add(oneShape.toShapeFX());

        Scene scene2 = new Scene(root2, 800, 450);

        secondStage.setScene(scene2);
        secondStage.setTitle("Circle inside");
        secondStage.show();
    }

    public static void main(String... args) {
        Application.launch(args);
    }
}