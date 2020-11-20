package com.cgoedefroit.td3_4.shape.sphere;

import com.cgoedefroit.td3_4.shape.AxesAlignedRectangle;
import com.cgoedefroit.td3_4.shape.AxesAlignedSquare;
import com.cgoedefroit.td3_4.shape.Triangle;
import com.cgoedefroit.td3_4.shape.elementary.Point2D;
import com.cgoedefroit.td3_4.shape.body.Polygone;
import com.cgoedefroit.td3_4.shape.Shape2D;
import javafx.scene.paint.Color;
import javafx.scene.shape.Shape;

import java.util.ArrayList;

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
        return this.centre.equals(p);
    }

    public boolean isInside(Polygone p) {
        if (p instanceof AxesAlignedSquare) {
            AxesAlignedSquare s = (AxesAlignedSquare) p;
            return // On top
                    s.inside(new Point2D(this.centre.getX(), this.centre.getY() + this.radius))
                            // On bottom
                            && s.inside(new Point2D(this.centre.getX(), this.centre.getY() - this.radius))
                            // On right
                            && s.inside(new Point2D(this.centre.getX() + this.radius, this.centre.getY()))
                            // On left
                            && s.inside(new Point2D(this.centre.getX() - this.radius, this.centre.getY()));
        } else if (p instanceof AxesAlignedRectangle) {
            AxesAlignedRectangle r = (AxesAlignedRectangle) p;
            return // On top
                    r.inside(new Point2D(this.centre.getX(), this.centre.getY() + this.radius))
                            // On bottom
                            && r.inside(new Point2D(this.centre.getX(), this.centre.getY() - this.radius))
                            // On right
                            && r.inside(new Point2D(this.centre.getX() + this.radius, this.centre.getY()))
                            // On left
                            && r.inside(new Point2D(this.centre.getX() - this.radius, this.centre.getY()));
        } else if (p instanceof Triangle) {
            Triangle t = (Triangle) p;
            return // On top
                    t.inside(new Point2D(this.centre.getX(), this.centre.getY() + this.radius))
                            // On bottom
                            && t.inside(new Point2D(this.centre.getX(), this.centre.getY() - this.radius))
                            // On right
                            && t.inside(new Point2D(this.centre.getX() + this.radius, this.centre.getY()))
                            // On left
                            && t.inside(new Point2D(this.centre.getX() - this.radius, this.centre.getY()));
        } else return false;
    }

    public boolean isInside(Circle c) {

        return  // On top
                c.inside(new Point2D(this.centre.getX(), this.centre.getY() + this.radius))
                        // On bottom
                        && c.inside(new Point2D(this.centre.getX(), this.centre.getY() - this.radius))
                        // On right
                        && c.inside(new Point2D(this.centre.getX() + this.radius, this.centre.getY()))
                        // On left
                        && c.inside(new Point2D(this.centre.getX() - this.radius, this.centre.getY()));
    }

    public boolean inside(Point2D p) {
        return p.distance(this.centre) <= radius;
    }

    public boolean inside(Polygone p) {
        ArrayList<Point2D> v;
        if (p instanceof AxesAlignedSquare) {
            AxesAlignedSquare s = (AxesAlignedSquare) p;
            v = s.vertices();
        } else if (p instanceof AxesAlignedRectangle) {
            AxesAlignedRectangle r = (AxesAlignedRectangle) p;
            v = r.vertices();
        } else if (p instanceof Triangle) {
            Triangle t = (Triangle) p;
            v = t.vertices();
        } else return false;

        for (Point2D point : v) if (!this.inside(point)) return false;
        return true;
    }

    public boolean inside(Circle c) {

        return  // On top
                this.inside(new Point2D(c.centre.getX(), c.centre.getY() + this.radius))
                        // On bottom
                        && this.inside(new Point2D(c.centre.getX(), c.centre.getY() - this.radius))
                        // On right
                        && this.inside(new Point2D(c.centre.getX() + this.radius, c.centre.getY()))
                        // On left
                        && this.inside(new Point2D(c.centre.getX() - this.radius, c.centre.getY()));
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