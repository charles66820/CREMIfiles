package com.cgoedefroit.tp3.shape.sphere;

import com.cgoedefroit.tp3.shape.elementary.Point2D;
import com.cgoedefroit.tp3.shape.body.Polygone;
import com.cgoedefroit.tp3.shape.Shape2D;
import javafx.scene.paint.Color;
import javafx.scene.shape.Shape;

public class Circle extends Shape2D {
    private final double radius;
    private final Point2D centre;

    public Circle(Point2D centre, double radius) {
        this(centre, radius, "undefine");
    }

    public Circle(Point2D centre, double radius, String name) {
        this.centre = new Point2D(centre);
        this.radius = radius;
        this.name = name;
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
        return new javafx.scene.shape.Circle(this.centre.getX(), this.centre.getY(), this.radius, Color.rgb(this.r, this.g, this.b, this.a));
    }

    @Override
    public String toString() {
        return "Circle ( " + this.name + ", " + this.radius + ", " + this.centre + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || this.getClass() != o.getClass()) return false;
        Circle circle = (Circle) o;
        return Double.compare(circle.radius, this.radius) == 0 &&
                this.centre.equals(circle.centre) &&
                this.r == circle.r &&
                this.g == circle.g &&
                this.b == circle.b;
    }

}