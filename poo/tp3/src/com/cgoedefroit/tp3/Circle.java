package com.cgoedefroit.tp3;

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

    @Override
    public void print() {
        System.out.println("Circle (" + this.radius + ", Point2D (" + this.centre.getX() + ", "  + this.centre.getY() + ")" + ")");
    }

    public boolean isInside(Point2D p) {
        return p.distance(this.centre) <= radius;
    }

    public String svg() {
        return "<circle cx='" + this.centre.getX() + "' cy='" + this.centre.getY() + "' r='" + this.radius + "' fill='rgb(" + this.r + "," + this.g + "," + this.b + ")' />";
    }
}