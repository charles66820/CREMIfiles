package com.cgoedefroit.tp3;

public class AxesAlignedSquare extends Shape2D {

    private Point2D pos;
    private int side;

    public AxesAlignedSquare(Point2D pos, int side) {
        this.pos = new Point2D(pos);
        this.side = side;
    }

    // Get / set
    public Point2D getPos() {
        return pos;
    }

    public void setPos(Point2D pos) {
        this.pos = pos;
    }

    public int getSide() {
        return side;
    }

    public void setSide(int side) {
        this.side = side;
    }

    // Methods
    public double area() {
        return this.side * this.side;
    }

    @Override
    public double perimeter() {
        return 4 * this.side;
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
        System.out.println("Square (" + this.side + ", Point2D (" + this.pos.getX() + ", "  + this.pos.getY() + ")" + ")");
    }

    public boolean isInside(Point2D p) {
        return (p.getX() >= this.pos.getX() && p.getX() < this.pos.getX() + this.side) &&
                (p.getY() >= this.pos.getY() && p.getY() < this.pos.getY() + this.side);
    }

    public String svg() {
        return "<rect x='" + this.pos.getX() + "' y='" + this.pos.getY() + "' width='" + this.side + "' height='" + this.side + "' stroke='rgb(" + this.r + "," + this.g + "," + this.b + ")' stroke-width='3' fill='#000000' fill-opacity='0' />";
    }
}
