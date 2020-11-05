package com.cgoedefroit.td1_2;

class TestPoint2D {
    public static void main(String[] args) {
        Point2D p = new Point2D(4.0, 3.0);
        p.print();
        p.move(2.0, 6.0);
        p.print();
        p.move(-2.0);
        p.print();
        Point2D p2 = new Point2D(6.0, 29.0);
        System.out.println(p.distance(p2));
    }
}