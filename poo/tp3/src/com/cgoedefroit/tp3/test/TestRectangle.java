package com.cgoedefroit.tp3.test;

import com.cgoedefroit.tp3.shape.AxesAlignedRectangle;
import com.cgoedefroit.tp3.shape.elementary.Point2D;

import java.io.FileWriter;
import java.io.IOException;

class TestRectangle {
    public static void main(String[] args) throws IOException {
        AxesAlignedRectangle r = new AxesAlignedRectangle(new Point2D(10.0, 10.0), 20, 40);
        System.out.println("Surface: " + r.area());
        System.out.println("Perimeter: " + r.perimeter());
        System.out.println("Test with point in x: 10 y: 20");
        System.out.println(r.isInside(new Point2D(10.0, 20.0)));
        System.out.println("Test with point in x: 20 y: 20");
        System.out.println(r.isInside(new Point2D(20.0, 20.0)));
        System.out.println("Test with point in x: 10 y: 10");
        System.out.println(r.isInside(new Point2D(10.0, 30.0)));
        System.out.println("Test with point in x: -10 y: 10");
        System.out.println(r.isInside(new Point2D(-10.0, 10.0)));
        r.print();
        r.translate(20, 10);
        r.print();
        r.translate(-10);
        r.print();

        r.setR(255);
        FileWriter out = new FileWriter("rectangle.svg");
        out.write("<?xml version='1.0' encoding='utf-8'?>\n");
        out.write("<svg xmlns='http://www.w3.org/2000/svg' version='1.1' width='100' height='100'>");
        out.write(r.svg());
        out.write("</svg>");
        out.close();
    }
}