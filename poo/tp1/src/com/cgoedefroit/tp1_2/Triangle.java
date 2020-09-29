package com.cgoedefroit.tp1_2;

public class Triangle {
    private final Point2D p1;
    private final Point2D p2;
    private final Point2D p3;

    private int r;
    private int g;
    private int b;

    public Triangle(Point2D p1, Point2D p2, Point2D p3) {
        this.p1 = p1;
        this.p2 = p2;
        this.p3 = p3;
    }

    public int getR() {
        return this.r;
    }

    public void setR(int v) {
        this.r = v;
    }

    public int getG() {
        return this.g;
    }

    public void setG(int v) {
        this.g = v;
    }

    public int getB() {
        return this.b;
    }

    public void setB(int v) {
        this.b = v;
    }

    public double surface() {
        return (this.p1.distance(p2) * this.p1.distance(p3)) / 2;
    }

    public boolean isInside(Point2D p) {
        return p1.distance(p) <= p1.distance(p2) &&
        p1.distance(p) <= p1.distance(p3) &&
        p2.distance(p) <= p2.distance(p1) &&
        p2.distance(p) <= p2.distance(p3) &&
        p3.distance(p) <= p3.distance(p1) &&
        p3.distance(p) <= p3.distance(p2);
    }

    public double perimeter() {
        return this.p1.distance(this.p2) + this.p2.distance(this.p3) + this.p3.distance(this.p1);
    }

    public void move(double dx, double dy) {
        this.p1.move(dx, dy);
        this.p2.move(dx, dy);
        this.p3.move(dx, dy);
    }

    public void move(double delta) {
        this.p1.move(delta);
        this.p2.move(delta);
        this.p3.move(delta);
    }

    public void print() {
        System.out.println("Triangle: pos x:" + this.p1.getX() + ", pos y: " + this.p1.getY());
    }

    public boolean isIsosceles() {
        return this.p1.distance(this.p2) == this.p2.distance(this.p3) || this.p1.distance(this.p2) == this.p3.distance(this.p1);
    }

    public String svg() {
        return "<polygon points='" + this.p1.getX() + "," + this.p1.getY() + " " + this.p2.getX() + "," + this.p2.getY() + " " + this.p3.getX() + "," + this.p3.getY() + "' stroke='rgb(" + this.r + "," + this.g + "," + this.b + ")' stroke-width='3' />";
    }
}