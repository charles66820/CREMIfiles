package com.cgoedefroit.td1_2;

public class AxesAlignedSquare {

    private Point2D pos;
    private int side;

    private int r;
    private int g;
    private int b;

    public AxesAlignedSquare(Point2D pos, int side) {
        this.pos = new Point2D(pos.getX(), pos.getY());;
        this.side = side;
    }

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

    public double surface() {
        return this.side * this.side;
    }

    public boolean isInside(Point2D p) {
        return (p.getX() >= this.pos.getX() && p.getX() < this.pos.getX() + this.side) &&
                (p.getY() >= this.pos.getY() && p.getY() < this.pos.getY() + this.side);
    }

    public double perimeter() {
        return 4 * this.side;
    }

    public void move(double dx, double dy) {
        this.pos.move(dx, dy);
    }

    public void move(double delta) {
        this.pos.move(delta);
    }

    public void print() {
        System.out.println("Square: pos x:" + this.pos.getX() + ", pos y: " + this.pos.getY());
    }

    public String svg() {
        return "<rect x='" + this.pos.getX() + "' y='" + this.pos.getY() + "' width='" + this.side + "' height='" + this.side + "' stroke='rgb(" + this.r + "," + this.g + "," + this.b + ")' stroke-width='3' />";
    }
}
