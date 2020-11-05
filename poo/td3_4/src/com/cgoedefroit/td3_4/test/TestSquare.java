package com.cgoedefroit.td3_4.test;

import com.cgoedefroit.td3_4.shape.AxesAlignedSquare;
import com.cgoedefroit.td3_4.shape.elementary.Point2D;

import java.io.FileWriter;
import java.io.IOException;

class TestSquare {
    public static void main(String[] args) throws IOException {
        AxesAlignedSquare s = new AxesAlignedSquare(new Point2D(10.0, 10.0), 20);
        System.out.println("Surface: " + s.area());
        System.out.println("Perimeter: " + s.perimeter());
        System.out.println("Test with point in x: 10 y: 20");
        System.out.println(s.inside(new Point2D(10.0, 20.0)));
        System.out.println("Test with point in x: 20 y: 20");
        System.out.println(s.inside(new Point2D(20.0, 20.0)));
        System.out.println("Test with point in x: 10 y: 10");
        System.out.println(s.inside(new Point2D(10.0, 30.0)));
        System.out.println("Test with point in x: -10 y: 10");
        System.out.println(s.inside(new Point2D(-10.0, 10.0)));
        s.print();
        s.translate(20, 10);
        s.print();
        s.translate(-10);
        s.print();

        s.setR(255);
        FileWriter out = new FileWriter("square.svg");
        out.write("<?xml version='1.0' encoding='utf-8'?>\n");
        out.write("<svg xmlns='http://www.w3.org/2000/svg' version='1.1' width='100' height='100'>");
        out.write(s.svg());
        out.write("</svg>");
        out.close();
    }
}