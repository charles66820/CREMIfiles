package com.cgoedefroit.td7;

import java.util.ArrayList;

public class Point2D {
    private double x, y;
    private final String name;

    public Point2D(double x, double y) {
        this(x, y, "undefine");
    }

    public Point2D(double x, double y, String name) {
        this.x = x;
        this.y = y;
        this.name = name;
    }

    public Point2D(Point2D p) {
        this(p.x, p.y, p.name);
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
    public void translate(double dx, double dy) {
        this.x += dx;
        this.y += dy;
    }

    public void translate(double d) {
        this.x += d;
        this.y += d;
    }

    public boolean inside(Point2D p) {
        return this.equals(p);
    }

    public double distance(Point2D p) {
        return Math.sqrt(Math.pow(p.x - this.x, 2) + Math.pow(p.y - this.y, 2));
    }

    @Override
    public String toString() {
        return "Point2D ( " + this.name + ", " + x + ", " + y + " )@" + this.hashCode();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || this.getClass() != o.getClass()) return false;
        Point2D point2D = (Point2D) o;
        return Double.compare(point2D.x, this.x) == 0 &&
                Double.compare(point2D.y, this.y) == 0;
    }
}
