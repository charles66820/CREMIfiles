package com.cgoedefroit.tp3;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public abstract class Shape2D {
    protected int r, g, b;

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

    // Methods
    public void translate(double dx, double dy) {
        throw new NotImplementedException();
    }

    public void translate(double delta) {
        translate(delta, delta);
    }

    public void print() {
        System.out.println("Shape2D");
    }
}