package com.cgoedefroit.tp3;

public class AxesAlignedRectangle extends Shape2D {

    private Point2D pos;
    private int height, width;

    public AxesAlignedRectangle(Point2D pos, int height, int width) {
        this.pos = new Point2D(pos);
        this.height = height;
        this.width = width;
    }

    // Get / set
    public Point2D getPos() {
        return pos;
    }

    public void setPos(Point2D pos) {
        this.pos = pos;
    }

    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public int getWidth() {
        return width;
    }

    public void setWidth(int width) {
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

    @Override
    public void print() {
        System.out.println("Rectangle (" + this.height + ", " + this.width + ", Point2D (" + this.pos.getX() + ", "  + this.pos.getY() + ")" + ")");
    }

    public boolean isInside(Point2D p) {
        return (p.getX() >= this.pos.getX() && p.getX() < this.pos.getX() + this.width) &&
                (p.getY() >= this.pos.getY() && p.getY() < this.pos.getY() + this.height);
    }

    public String svg() {
        return "<rect x='" + this.pos.getX() + "' y='" + this.pos.getY() + "' width='" + this.width + "' height='" + this.height + "' stroke='rgb(" + super.r + "," + super.g + "," + super.b + ")' stroke-width='3' />";
    }
}
