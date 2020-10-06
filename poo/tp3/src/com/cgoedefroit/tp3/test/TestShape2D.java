package com.cgoedefroit.tp3.test;

import com.cgoedefroit.tp3.shape.elementary.Point2D;

class TestShape2D {
    public static void main(String[] args) {
        Point2D p = new Point2D(1, 2);
        p.print();
        p.translate(5);
        p.print();
        Point2D p2 = new Point2D(4, 4);
        System.out.println(p.distance(p2));
        System.out.println("Le premier point est " + p + " et le deuxieme " + p2);

        Point2D D = new Point2D(1.0, 1.0, "D");
        Point2D E = new Point2D(1.0, 1.0, "E");
        if (D.equals(E)) {
            System.out.println("Les points sont égaux");
        }

    }
}