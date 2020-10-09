package com.cgoedefroit.tp3.util;

import com.cgoedefroit.tp3.shape.AxesAlignedRectangle;
import com.cgoedefroit.tp3.shape.AxesAlignedSquare;
import com.cgoedefroit.tp3.shape.Shape2D;
import com.cgoedefroit.tp3.shape.Triangle;
import com.cgoedefroit.tp3.shape.elementary.Point2D;
import com.cgoedefroit.tp3.shape.sphere.Circle;

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
                    r.setA(randomizer.nextDouble());
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
                    s.setA(randomizer.nextDouble());
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
                    c.setA(randomizer.nextDouble());
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
                    t.setA(randomizer.nextDouble());
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

        if (c.inside(s)) c.setG(255);
        else c.setR(255);

        // Test with circle inside square
        AxesAlignedSquare s2 = new AxesAlignedSquare(new Point2D(30 + (s.getPos().getX() * 2) + 30, 30), s.getSide(), "square ");
        s2.setB(255);
        s2.setA(0.7);
        sList.add(s2);

        Circle c2 = new Circle(new Point2D(s2.getPos().getX() + (s2.getSide() / 2), 30 + (s2.getSide() / 2)), 20, "circle ");
        c2.setA(0.7);
        sList.add(c2);

        if (c2.inside(s)) c2.setG(255);
        else c2.setR(255);
        return sList;
    }
}
