package com.cgoedefroit.tp1_2;

public class Point2D {
    private double x;
    private double y;

    public Point2D() {
    }

    public Point2D(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public double getX() {
        return this.x;
    }

    public void setX(double v) {
        this.x = v;
    }

    public double getY() {
        return this.y;
    }

    public void setY(double v) {
        this.y = v;
    }

    public void print() {
        System.out.println("Point: x: " + this.x + " y: " + this.y);
    }

    public void move(double dx, double dy) {
        this.x += dx;
        this.y += dy;
    }

    public void move(double delta) {
        this.x += delta;
        this.y += delta;
    }

    public double distance(Point2D p) {
        return Math.sqrt(Math.pow(p.getX() - this.x, 2) + Math.pow(p.getY() - this.y, 2));
    }

}