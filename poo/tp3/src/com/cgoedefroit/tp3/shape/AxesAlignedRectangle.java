package com.cgoedefroit.tp3.shape;

import com.cgoedefroit.tp3.shape.elementary.Point2D;
import com.cgoedefroit.tp3.shape.body.Polygone;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.scene.shape.Shape;

import java.util.ArrayList;
import java.util.Objects;

public class AxesAlignedRectangle extends Polygone {

    protected Point2D pos;
    private double height, width;

    public AxesAlignedRectangle(Point2D pos, double height, double width) {
        this(pos, height, width, "undefine");
    }

    public AxesAlignedRectangle(Point2D pos, double height, double width, String name) {
        this.pos = new Point2D(pos);
        this.height = height;
        this.width = width;
        this.name = name;
    }

    // Get / set
    public Point2D getPos() {
        return pos;
    }

    public void setPos(Point2D pos) {
        this.pos = pos;
    }

    public double getHeight() {
        return height;
    }

    public void setHeight(double height) {
        this.height = height;
    }

    public double getWidth() {
        return width;
    }

    public void setWidth(double width) {
        this.width = width;
    }

    // Methods
    @Override
    public double area() {
        return this.height * this.width;
    }

    @Override
    public double perimeter() {
        return (2 * this.width) + (2 * this.height);
    }

    @Override
    public void translate(double dx, double dy) {
        this.pos.translate(dx, dy);
    }

    @Override
    public void translate(double delta) {
        this.pos.translate(delta);
    }

    public boolean isInside(Point2D p) {
        return (p.getX() >= this.pos.getX() && p.getX() < this.pos.getX() + this.width) &&
                (p.getY() >= this.pos.getY() && p.getY() < this.pos.getY() + this.height);
    }

    @Override
    public ArrayList<Point2D> vertices() {
        ArrayList<Point2D> vList = new ArrayList<Point2D>();
        vList.add(pos);
        vList.add(new Point2D(pos.getX(), pos.getY() + this.height));
        vList.add(new Point2D(pos.getX() + this.width, pos.getY() + this.height));
        vList.add(new Point2D(pos.getX() + this.width, pos.getY()));
        return vList;
    }

    public String svg() {
        return "<rect x='" + this.pos.getX() + "' y='" + this.pos.getY() + "' width='" + this.width + "' height='" + this.height + "' stroke='rgb(" + this.r + "," + this.g + "," + this.b + ")' stroke-width='3' fill='#000000' fill-opacity='0' />";
    }

    public Shape toShapeFX() {
        Rectangle rectangle = new Rectangle(this.pos.getX(), this.pos.getY(), this.pos.getX() + this.width, this.pos.getY() + this.height);
        rectangle.setFill(Color.rgb(this.r, this.g, this.b, this.a));
        return rectangle;
    }

    @Override
    public String toString() {
        return "Rectangle ( " + this.name + ", " + this.height + ", " + this.width + ", " + this.pos + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || this.getClass() != o.getClass()) return false;
        AxesAlignedRectangle that = (AxesAlignedRectangle) o;
        return this.height == that.height &&
                this.width == that.width &&
                this.pos.equals(that.pos) &&
                this.r == that.r &&
                this.g == that.g &&
                this.b == that.b;
    }
}
