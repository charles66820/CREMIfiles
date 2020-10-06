package com.cgoedefroit.tp3.shape.elementary;

import com.cgoedefroit.tp3.shape.Shape2D;

import java.util.Objects;

public class Point2D extends Shape2D {
    private double x, y;

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
        return "Point2D ( " + this.name + ", " + x + ", " + y + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || this.getClass() != o.getClass()) return false;
        Point2D point2D = (Point2D) o;
        return Double.compare(point2D.x, this.x) == 0 &&
                Double.compare(point2D.y, this.y) == 0 &&
                this.r == point2D.r &&
                this.g == point2D.g &&
                this.b == point2D.b;
    }
}