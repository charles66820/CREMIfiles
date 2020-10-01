package com.cgoedefroit.tp3;

import java.io.FileWriter;
import java.io.IOException;

class TestTriangle {
    public static void main(String[] args) throws IOException {
        Triangle t = new Triangle(new Point2D(0.0, 0.0), new Point2D(0.0, 40.0), new Point2D(20.0, 40.0));
        System.out.println("Surface: " + t.area());
        System.out.println("Perimeter: " + t.perimeter());
        System.out.println("Test with point in x: 10 y: 20");
        System.out.println(t.isInside(new Point2D(10.0, 20.0)));
        System.out.println("Test with point in x: 20 y: 20");
        System.out.println(t.isInside(new Point2D(20.0, 20.0)));
        System.out.println("Test with point in x: 10 y: 10");
        System.out.println(t.isInside(new Point2D(10.0, 30.0)));
        System.out.println("Test with point in x: -10 y: 10");
        System.out.println(t.isInside(new Point2D(-10.0, 10.0)));
        t.print();
        t.translate(20, 10);
        t.print();
        t.translate(-10);
        t.print();

        FileWriter out = new FileWriter("triangle.svg");
        out.write("<?xml version='1.0' encoding='utf-8'?>\n");
        out.write("<svg xmlns='http://www.w3.org/2000/svg' version='1.1' width='100' height='100'>");
        out.write(t.svg());
        out.write("</svg>");
        out.close();
    }
}