package com.cgoedefroit.tp3;

public class Point2D extends Shape2D {
    private double x, y;

    public Point2D(double x, double y)  {
        this.x = x;
        this.y = y;
    }

    public Point2D(Point2D p){
        this(p.x, p.y);
    }

    // Get / set
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

    // Methods
    @Override
    public void translate(double dx, double dy) {
        this.x += dx;
        this.y += dy;
    }
    
    public double distance(Point2D p) {
        return Math.sqrt(Math.pow(p.x - this.x, 2) + Math.pow(p.y - this.y, 2));
    }

    @Override
    public String toString() {
        return "Point2D (" + x + ", "  + y + ")";
    }
}