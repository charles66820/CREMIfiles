package com.cgoedefroit.tp3_4.test;

import com.cgoedefroit.tp3_4.shape.Shape2D;
import com.cgoedefroit.tp3_4.shape.Triangle;
import com.cgoedefroit.tp3_4.shape.elementary.Point2D;
import com.cgoedefroit.tp3_4.util.tools;

import java.util.ArrayList;

class TestShape2D {

    public static void main(String[] args) {

        // Test Point2D
        System.out.println("======Test Point2D======");
        Point2D p1 = new Point2D(1, 2);
        p1.print();
        p1.translate(5);
        p1.print();
        Point2D p2 = new Point2D(4, 4);
        System.out.println(p1.distance(p2));
        System.out.println("Le premier point est " + p1 + " et le deuxieme " + p2);

        // Test polymorphisme
        System.out.println("======Test polymorphisme======");
        Shape2D s = new Point2D(1, 2);
        s.print();

        Point2D A = new Point2D(0, 0);
        Point2D B = new Point2D(5, 0);
        Point2D C = new Point2D(0, 5);
        s = new Triangle(A, B, C);
        s.print();

        // Table of Shape2D
        System.out.println("======Test Table of Shape2D======");
        // Randowm ArrayList of Shape2D
        ArrayList<Shape2D> sList = tools.randShape2DList(20, 30, 10, 10, 50, 50);

        for (Shape2D oneShape : sList) oneShape.print();

        // Test equals
        System.out.println("======Test equals======");
        Point2D D = new Point2D(1.0, 1.0, "D");
        Point2D E = new Point2D(1.0, 1.0, "E");
        if (D.equals(E)) {
            System.out.println("Les points sont égaux");
        }

        // One point
        System.out.println("======Show shape with point inside======");
        Point2D p = new Point2D(20, 25);
        // Find shape with p inside in sList
        for (Shape2D oneShape : sList) if(oneShape.inside(p)) oneShape.print();

    }
}