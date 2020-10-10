package com.cgoedefroit.tp3_4app;

import com.cgoedefroit.tp3_4.shape.Shape2D;
import com.cgoedefroit.tp3_4.shape.elementary.Point2D;
import com.cgoedefroit.tp3_4.util.tools;
import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.util.ArrayList;

public class tp3_4app extends Application {

    public void start(Stage stage) throws Exception {

        Group root = new Group();
        ArrayList<Shape2D> sList = tools.randShape2DList(40, 80, 200, 100, 800, 450);

        for (Shape2D oneShape : sList) root.getChildren().add(oneShape.toShapeFX());

        stage.setScene(new Scene(root, 800, 450));
        stage.setTitle("Rand Shape2D List");
        stage.show();

        Stage secondStage = new Stage();
        Group root2 = new Group();
        ArrayList<Shape2D> sList2 = tools.circleInPolygoneList();

        for (Shape2D oneShape : sList2) root2.getChildren().add(oneShape.toShapeFX());

        secondStage.setScene(new Scene(root2, 800, 450));
        secondStage.setTitle("Circle inside");
        secondStage.show();

        Stage therdStage = new Stage();
        Group root3 = new Group();

        Point2D p = new Point2D(500, 300);
        // Find shape with p inside in sList
        for (Shape2D oneShape : sList) if(oneShape.inside(p)) root3.getChildren().add(oneShape.toShapeFX());

        therdStage.setScene(new Scene(root3, 800, 450));
        therdStage.setTitle("Show shape with point inside");
        therdStage.show();
    }

    public static void main(String... args) {
        Application.launch(args);
    }
}