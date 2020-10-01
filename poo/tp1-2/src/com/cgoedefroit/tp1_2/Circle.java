package com.cgoedefroit.tp1_2;

public class Circle {
    private final double radius;
    private final Point2D centre;

    private int r;
    private int g;
    private int b;

    public Circle(Point2D centre, double radius) {
        this.centre = new Point2D(centre.getX(), centre.getY());
        this.radius = radius;
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
        return Math.PI * Math.sqrt(this.radius);
    }

    public boolean isInside(Point2D p) {
        return p.distance(this.centre) <= radius;
    }

    public double perimeter() {
        return 2 * Math.PI * this.radius;
    }

    public void move(double dx, double dy) {
        this.centre.move(dx, dy);
    }

    public void move(double delta) {
        this.centre.move(delta);
    }

    public void print() { System.out.println("Circle: centre x:" + this.centre.getX() + ", centre y: " + this.centre.getY()); }

    public String svg() {
        return "<circle cx='" + this.centre.getX() + "' cy='" + this.centre.getY() + "' r='" + this.radius + "' fill='rgb(" + this.r + "," + this.g + "," + this.b + ")' />";
    }
}