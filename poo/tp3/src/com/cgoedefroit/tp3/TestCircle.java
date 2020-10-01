package com.cgoedefroit.tp3;

import java.io.FileWriter;
import java.io.IOException;

class TestCircle {
    public static void main(String[] args) throws IOException {
        Circle c = new Circle(new Point2D(40.0, 30.0), 40);
        System.out.println("Surface: " + c.area());
        System.out.println("Perimeter: " + c.perimeter());
        System.out.println(c.isInside(new Point2D(20.0, 20.0)));
        System.out.println(c.isInside(new Point2D(90.0, 80.0)));
        c.print();
        c.translate(2.0, 6.0);
        c.print();
        c.translate(-2.0);
        c.print();

        c.setR(255);
        FileWriter out = new FileWriter("circle.svg");
        out.write("<?xml version='1.0' encoding='utf-8'?>\n");
        out.write("<svg xmlns='http://www.w3.org/2000/svg' version='1.1' width='100' height='100'>");
        out.write(c.svg());
        out.write("</svg>");
        out.close();
    }
}