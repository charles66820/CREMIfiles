package com.cgoedefroit.tp3_4.shape;

import com.cgoedefroit.tp3_4.shape.elementary.Point2D;
import com.cgoedefroit.tp3_4.shape.body.Polygone;
import javafx.scene.paint.Color;
import javafx.scene.shape.Polygon;
import javafx.scene.shape.Shape;

import java.util.ArrayList;

public class Triangle extends Polygone {
    private final Point2D p1;
    private final Point2D p2;
    private final Point2D p3;

    public Triangle(Point2D p1, Point2D p2, Point2D p3) {
        this(p1, p2, p3, "undefine");
    }

    public Triangle(Point2D p1, Point2D p2, Point2D p3, String name) {
        this.p1 = new Point2D(p1);
        this.p2 = new Point2D(p2);
        this.p3 = new Point2D(p3);
        this.name = name;
    }

    // Methods
    @Override
    public double area() {
        return (this.p1.distance(p2) * this.p1.distance(p3)) / 2;
    }

    @Override
    public double perimeter() {
        return this.p1.distance(this.p2) + this.p2.distance(this.p3) + this.p3.distance(this.p1);
    }

    @Override
    public void translate(double dx, double dy) {
        this.p1.translate(dx, dy);
        this.p2.translate(dx, dy);
        this.p3.translate(dx, dy);
    }

    @Override
    public void translate(double delta) {
        this.p1.translate(delta);
        this.p2.translate(delta);
        this.p3.translate(delta);
    }

    @Override
    public boolean inside(Point2D p) {
        return p1.distance(p) <= p1.distance(p2) &&
        p1.distance(p) <= p1.distance(p3) &&
        p2.distance(p) <= p2.distance(p1) &&
        p2.distance(p) <= p2.distance(p3) &&
        p3.distance(p) <= p3.distance(p1) &&
        p3.distance(p) <= p3.distance(p2);
    }

    public boolean isIsosceles() {
        return this.p1.distance(this.p2) == this.p2.distance(this.p3) || this.p1.distance(this.p2) == this.p3.distance(this.p1);
    }

    @Override
    public ArrayList<Point2D> vertices() {
        ArrayList<Point2D> vList = new ArrayList<Point2D>();
        vList.add(p1);
        vList.add(p2);
        vList.add(p3);
        return vList;
    }

    public String svg() {
        return "<polygon points='" + this.p1.getX() + "," + this.p1.getY() + " " + this.p2.getX() + "," + this.p2.getY() + " " + this.p3.getX() + "," + this.p3.getY() + "' stroke='rgb(" + this.r + "," + this.g + "," + this.b + ")' stroke-width='3' fill='#000000' fill-opacity='0' />";
    }

    public Shape toShapeFX() {
        Polygon polygon = new Polygon(this.p1.getX(), this.p1.getY(), this.p2.getX(), this.p2.getY(), this.p3.getX(), this.p3.getY());
        polygon.setFill(Color.rgb(this.r, this.g, this.b, this.a));
        return polygon;
    }

    @Override
    public String toString() {
        return "Triangle ( " + this.name + ", " + this.p1 + ", " +  this.p2 + ", " + this.p3 + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Triangle triangle = (Triangle) o;
        return this.p1.equals(triangle.p1) &&
                this.p2.equals(triangle.p2) &&
                this.p3.equals(triangle.p3) &&
                this.r == triangle.r &&
                this.g == triangle.g &&
                this.b == triangle.b;
    }
}