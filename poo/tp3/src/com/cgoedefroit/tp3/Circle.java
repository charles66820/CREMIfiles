package com.cgoedefroit.tp3;

import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.scene.shape.Shape;

public class Circle extends Shape2D {
    private final double radius;
    private final Point2D centre;

    public Circle(Point2D centre, double radius) {
        this.centre = new Point2D(centre);
        this.radius = radius;
    }

    // Methods
    @Override
    public double area() {
        return Math.PI * Math.sqrt(this.radius);
    }

    @Override
    public double perimeter() {
        return 2 * Math.PI * this.radius;
    }

    @Override
    public void translate(double dx, double dy) {
        this.centre.translate(dx, dy);
    }

    @Override
    public void translate(double delta) {
        this.centre.translate(delta);
    }

    public boolean isInside(Point2D p) {
        return p.distance(this.centre) <= radius;
    }

    boolean inside(Point2D p) {
        // TODO: implement this
        return false;
    }
    boolean inside(Polygone p) {
        // TODO: implement this
        // vertices and inside(Point2D p)
        return false;
    }
    boolean inside(Circle c) {
        // TODO: implement this
        return false;
    }

    public String svg() {
        return "<circle cx='" + this.centre.getX() + "' cy='" + this.centre.getY() + "' r='" + this.radius + "' stroke='rgb(" + this.r + "," + this.g + "," + this.b + ")' stroke-width='3' fill='#000000' fill-opacity='0' />";
    }

    public Shape toShapeFX() {
        return new javafx.scene.shape.Circle(this.centre.getX(), this.centre.getY(), this.radius, Color.rgb(this.r,this.g,this.b,1.0));
    }

    @Override
    public String toString() {
        return "Circle (" + this.radius + ", Point2D (" + this.centre.getX() + ", "  + this.centre.getY() + ")" + ")";
    }
}