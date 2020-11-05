package com.cgoedefroit.td3_4.util;

import com.cgoedefroit.td3_4.shape.AxesAlignedRectangle;
import com.cgoedefroit.td3_4.shape.AxesAlignedSquare;
import com.cgoedefroit.td3_4.shape.Shape2D;
import com.cgoedefroit.td3_4.shape.Triangle;
import com.cgoedefroit.td3_4.shape.elementary.Point2D;
import com.cgoedefroit.td3_4.shape.sphere.Circle;

import java.util.ArrayList;
import java.util.Random;

public class tools {

    private final static Random randomizer = new Random();

    public static ArrayList<Shape2D> randShape2DList(int nbMin, int nbMax, double startX, double startY, double stopX, double stopY) {

        ArrayList<Shape2D> sList = new ArrayList<>();

        // Add some shapes
        for (int i = 0; i < randomizer.nextInt(nbMax - nbMin + 1) + nbMin; i++) {
            Point2D p = new Point2D(
                    ((stopX - startX) * randomizer.nextDouble() + startX),
                    ((stopY - startY) * randomizer.nextDouble() + startY),
                    "point " + i
            );

            final boolean widthIsSmall = stopY - startY == Math.min(stopY - startY, stopX - startX);
            switch (randomizer.nextInt(5 - 1 + 1) + 1) {
                case 1:
                    // Add some point
                    sList.add(p);
                    break;
                case 2:
                    // Add some rectangle
                    AxesAlignedRectangle r = new AxesAlignedRectangle(
                            p,
                            ((stopY - startY) * randomizer.nextDouble() + startY) - p.getY(),
                            ((stopX - startX) * randomizer.nextDouble() + startX) - p.getX(),
                            "rectangle " + i
                    );
                    r.setR(randomizer.nextInt(256));
                    r.setG(randomizer.nextInt(256));
                    r.setB(randomizer.nextInt(256));
                    r.setA(((1 - 0.4) * randomizer.nextDouble()) + 0.4);
                    sList.add(r);
                    break;
                case 3:
                    // Add some square
                    AxesAlignedSquare s = new AxesAlignedSquare(
                            p,
                            widthIsSmall ? ((stopY - startY) * randomizer.nextDouble() + startY) - p.getY()
                                    : ((stopX - startX) * randomizer.nextDouble() + startX) - p.getX(),
                            "square " + i
                    );
                    s.setR(randomizer.nextInt(256));
                    s.setG(randomizer.nextInt(256));
                    s.setB(randomizer.nextInt(256));
                    s.setA(((1 - 0.4) * randomizer.nextDouble()) + 0.4);
                    sList.add(s);
                    break;
                case 4:
                    // Add some circle
                    Circle c = new Circle(
                            p,
                            widthIsSmall ? ((stopY - startY) * randomizer.nextDouble() + startY) - p.getY()
                                    : ((stopX - startX) * randomizer.nextDouble() + startX) - p.getX(),
                            "circle " + i
                    );
                    c.setR(randomizer.nextInt(256));
                    c.setG(randomizer.nextInt(256));
                    c.setB(randomizer.nextInt(256));
                    c.setA(((1 - 0.4) * randomizer.nextDouble()) + 0.4);
                    sList.add(c);
                    break;
                case 5:
                    Point2D p2 = new Point2D(
                            ((stopX - startX) * randomizer.nextDouble() + startX),
                            ((stopY - startY) * randomizer.nextDouble() + startY)
                    );
                    Point2D p3 = new Point2D(
                            ((stopX - startX) * randomizer.nextDouble() + startX),
                            ((stopY - startY) * randomizer.nextDouble() + startY)
                    );
                    // Add some triangle
                    Triangle t = new Triangle(p, p2, p3, "triangle " + i);
                    t.setR(randomizer.nextInt(256));
                    t.setG(randomizer.nextInt(256));
                    t.setB(randomizer.nextInt(256));
                    t.setA(((1 - 0.4) * randomizer.nextDouble()) + 0.4);
                    sList.add(t);
                    break;
            }
        }

        return sList;
    }

    public static ArrayList<Shape2D> circleInPolygoneList() {
        ArrayList<Shape2D> sList = new ArrayList<>();
        // Test with circle not inside square
        AxesAlignedSquare s = new AxesAlignedSquare(new Point2D(30, 30), 60, "square ");
        s.setB(255);
        s.setA(0.7);
        sList.add(s);

        Circle c = new Circle(new Point2D(30, 30), 20, "circle ");
        c.setA(0.7);
        sList.add(c);

        c.print();
        s.print();
        if (c.inside(s)) {
            c.setG(255);
            System.out.println("Circle 1 in square 1");
        } else {
            c.setR(255);
            System.out.println("Circle 1 not in square 1");
        }

        // Test with circle inside square
        AxesAlignedSquare s2 = new AxesAlignedSquare(new Point2D(30 + s.getSide() + 30, 30), s.getSide(), "square ");
        s2.setB(255);
        s2.setA(0.7);
        sList.add(s2);

        Circle c2 = new Circle(new Point2D(s2.getPos().getX() + (s2.getSide() / 2), 30 + (s2.getSide() / 2)), 20, "circle ");
        c2.setA(0.7);
        sList.add(c2);

        c2.print();
        s2.print();
        if (c2.inside(s2)) {
            c2.setG(255);
            System.out.println("Circle 2 in square 2");
        } else {
            c2.setR(255);
            System.out.println("Circle 2 not in square 2");
        }

        // Test with circle not inside rectangle
        AxesAlignedRectangle r = new AxesAlignedRectangle(new Point2D(30, 120), 60, 100, "rectangle ");
        r.setB(255);
        r.setA(0.7);
        sList.add(r);

        Circle c3 = new Circle(new Point2D(30, 120), 20, "circle ");
        c3.setA(0.7);
        sList.add(c3);

        c3.print();
        r.print();
        if (c3.inside(r)) {
            c3.setG(255);
            System.out.println("Circle 1 in rectangle 1");
        } else {
            c3.setR(255);
            System.out.println("Circle 1 not in rectangle 1");
        }

        // Test with circle inside rectangle
        AxesAlignedRectangle r2 = new AxesAlignedRectangle(new Point2D(30 * 3 + r.getHeight(), 120), r.getHeight(), r.getWidth(), "rectangle ");
        r2.setB(255);
        r2.setA(0.7);
        sList.add(r2);

        Circle c4 = new Circle(new Point2D(r2.getPos().getX() + (r2.getWidth() / 2), 120 + (r2.getHeight() / 2)), 20, "circle ");
        c4.setA(0.7);
        sList.add(c4);

        c4.print();
        r2.print();
        if (c4.inside(r2)) {
            c4.setG(255);
            System.out.println("Circle 2 in rectangle 2");
        } else {
            c4.setR(255);
            System.out.println("Circle 2 not in rectangle 2");
        }

        // Test with circle not inside triangle
        Triangle t = new Triangle(new Point2D(30, 270), new Point2D(90, 210), new Point2D(150, 270), "triangle");
        t.setB(255);
        t.setA(0.7);
        sList.add(t);

        Circle c5 = new Circle(new Point2D(70, 240), 20, "circle ");
        c5.setA(0.7);
        sList.add(c5);

        c5.print();
        t.print();
        if (c5.inside(t)) {
            c5.setG(255);
            System.out.println("Circle 1 in triangle 1");
        } else {
            c5.setR(255);
            System.out.println("Circle 1 not in triangle 1");
        }

        // Test with circle inside triangle
        Triangle t2 = new Triangle(new Point2D(30 + 120, 270), new Point2D(90 + 120, 210), new Point2D(150 + 120, 270), "triangle ");
        t2.setB(255);
        t2.setA(0.7);
        sList.add(t2);

        Circle c6 = new Circle(new Point2D(210, 245), 20, "circle ");
        c6.setA(0.7);
        sList.add(c6);

        c6.print();
        t2.print();
        if (c6.inside(t2)) {
            c6.setG(255);
            System.out.println("Circle 2 in triangle 2");
        } else {
            c6.setR(255);
            System.out.println("Circle 2 not in triangle 2");
        }

        // Test with circle not inside circle
        Circle c7 = new Circle(new Point2D(90, 330), 20, "circle ");
        c7.setB(255);
        c7.setA(0.7);
        sList.add(c7);

        Circle c8 = new Circle(new Point2D(80, 320), 15, "circle ");
        c8.setA(0.7);
        sList.add(c8);

        c8.print();
        c7.print();
        if (c8.inside(c7)) {
            c8.setG(255);
            System.out.println("Circle 1 in circle 1");
        } else {
            c8.setR(255);
            System.out.println("Circle 1 not in circle 1");
        }

        // Test with circle inside circle
        Circle c9 = new Circle(new Point2D(210, 330), 20, "circle ");
        c9.setB(255);
        c9.setA(0.7);
        sList.add(c9);

        Circle c10 = new Circle(new Point2D(210, 330), 15, "circle ");
        c10.setA(0.7);
        sList.add(c10);

        c10.print();
        c9.print();
        if (c10.inside(c9)) {
            c10.setG(255);
            System.out.println("Circle 2 in circle 2");
        } else {
            c10.setR(255);
            System.out.println("Circle 2 not in circle 2");
        }

        return sList;
    }
}
