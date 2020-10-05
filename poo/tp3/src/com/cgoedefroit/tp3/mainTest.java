package com.cgoedefroit.tp3;

import java.util.ArrayList;

public class mainTest {
    public static void main(String[] args) {
        Shape2D s = new Point2D(1, 2);
        s.print();

        Point2D A = new Point2D(0, 0);
        Point2D B = new Point2D(5, 0);
        Point2D C = new Point2D(0, 5);
        s = new Triangle(A, B, C);
        s.print();

        // List
        ArrayList<Shape2D> sList = new ArrayList<Shape2D>();
        // Add points
        sList.add(A);
        sList.add(B);
        sList.add(C);
        // Add triangle
        sList.add(s);
        // Add circle
        sList.add(new Circle(A, 25));
        // Add square
        sList.add(new AxesAlignedSquare(B, 20));
        // Add rectangle
        sList.add(new AxesAlignedRectangle(C, 80, 40));

        for (Shape2D oneShape : sList) oneShape.print();
    }
}
