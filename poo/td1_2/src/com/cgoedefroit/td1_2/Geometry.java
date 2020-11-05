package com.cgoedefroit.td1_2;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

class Geometry {

  public static Scanner clavier = new Scanner(System.in);

  public static void main(String[] args) throws IOException {
    Triangle t = new Triangle(Geometry.askPoint(), Geometry.askPoint(), Geometry.askPoint());
    System.out.println("Perimeter : " +  t.perimeter());
    System.out.println("Isosceles : " + t.isIsosceles());
    t.setR(255);
    t.setG(255);

    FileWriter out = new FileWriter("triangle.svg");
		out.write("<?xml version='1.0' encoding='utf-8'?>\n");
		out.write("<svg xmlns='http://www.w3.org/2000/svg' version='1.1' width='100' height='100'>");
		out.write(t.svg());
		out.write("</svg>");
		out.close();
  }

  private static Point2D askPoint() {
    Point2D p = new Point2D();
    System.out.println("Construction d'un nouveau point");
    System.out.println("Veuillez entrer x : ");
    p.setX(clavier.nextDouble());
    System.out.println("Veuillez entrer y : ");
    p.setY(clavier.nextDouble());
    return p;
  }
}