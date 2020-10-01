package com.cgoedefroit.tp3;

public class Triangle extends Shape2D {
    private final Point2D p1;
    private final Point2D p2;
    private final Point2D p3;

    public Triangle(Point2D p1, Point2D p2, Point2D p3) {
        this.p1 = new Point2D(p1);
        this.p2 = new Point2D(p2);
        this.p3 = new Point2D(p3);
    }

    // Methods
    @Override
    public double area() {
        return (this.p1.distance(p2) * this.p1.distance(p3)) / 2;
    }

    @Override
    public double perimeter() {
        return this.p1.distance(this.p2) + this.p2.distance(this.p3) + this.p3.distance(this.p1);
    }

    @Override
    public void translate(double dx, double dy) {
        this.p1.translate(dx, dy);
        this.p2.translate(dx, dy);
        this.p3.translate(dx, dy);
    }

    @Override
    public void translate(double delta) {
        this.p1.translate(delta);
        this.p2.translate(delta);
        this.p3.translate(delta);
    }

    @Override
    public void print() {
        System.out.println("Triangle ( Point2D (" + this.p1.getX() + ", "  + this.p1.getY() + ")" + ", Point2D (" + this.p2.getX() + ", "  + this.p2.getY() + ")" + ", Point2D (" + this.p3.getX() + ", "  + this.p3.getY() + ")" + ")");
    }

    public boolean isInside(Point2D p) {
        return p1.distance(p) <= p1.distance(p2) &&
        p1.distance(p) <= p1.distance(p3) &&
        p2.distance(p) <= p2.distance(p1) &&
        p2.distance(p) <= p2.distance(p3) &&
        p3.distance(p) <= p3.distance(p1) &&
        p3.distance(p) <= p3.distance(p2);
    }

    public boolean isIsosceles() {
        return this.p1.distance(this.p2) == this.p2.distance(this.p3) || this.p1.distance(this.p2) == this.p3.distance(this.p1);
    }

    public String svg() {
        return "<polygon points='" + this.p1.getX() + "," + this.p1.getY() + " " + this.p2.getX() + "," + this.p2.getY() + " " + this.p3.getX() + "," + this.p3.getY() + "' stroke='rgb(" + this.r + "," + this.g + "," + this.b + ")' stroke-width='3' />";
    }
}