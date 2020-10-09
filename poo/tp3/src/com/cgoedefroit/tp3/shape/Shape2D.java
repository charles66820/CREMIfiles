package com.cgoedefroit.tp3.shape;

import com.cgoedefroit.tp3.shape.elementary.Point2D;
import javafx.scene.shape.Shape;

public abstract class Shape2D {

    protected String name;

    protected int r, g, b;
    protected double a;

    public double area() {
        return 0;
    }

    public double perimeter() {
        return 0;
    }

    // Colors
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

    public double getA() {
        return this.a;
    }

    public void setA(double v) {
        this.a = v;
    }

    // Methods
    public abstract void translate(double dx, double dy);

    public void translate(double delta) {
        translate(delta, delta);
    }

    public abstract boolean isInside(Point2D p);

    public void print() {
        System.out.println(this);
    }

    @Override
    public String toString() {
        return "Shape2D ( " + this.name + " )";
    }

    public abstract Shape toShapeFX();
}
