package com.cgoedefroit.tp3;

class TestShape2D {
    public static void main(String[] args) {
        Point2D p = new Point2D(1, 2);
        p.print();
        p.translate(5);
        p.print();
        Point2D p2 = new Point2D(4, 4);
        System.out.println(p.distance(p2));
    }
}