package com.cgoedefroit.tp1_2;

public class Rectangle {

    private Point2D pos;
    private int height;
    private int width;

    private int r;
    private int g;
    private int b;

    Rectangle(Point2D pos, int height, int width) {
        this.pos = pos;
        this.height = height;
        this.width = width;
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

    public double surface() {
        return this.height * this.width;
    }

    public boolean isInside(Point2D p) {
        return (p.getX() >= this.pos.getX() && p.getX() < this.pos.getX() + this.width) &&
                (p.getY() >= this.pos.getY() && p.getY() < this.pos.getY() + this.height);
    }

    public double perimeter() {
        return (2 * this.width) + (2 * this.height);
    }

    public void move(double dx, double dy) {
        this.pos.move(dx, dy);
    }

    public void move(double delta) {
        this.pos.move(delta);
    }

    public void print() {
        System.out.println("Rectangle : pos x:" + this.pos.getX() + ", pos y: " + this.pos.getY());
    }

    public String svg() {
        return "<rect x='" + this.pos.getX() + "' y='" + this.pos.getY() + "' width='" + this.width + "' height='" + this.height + "' stroke='rgb(" + this.r + "," + this.g + "," + this.b + ")' stroke-width='3' />";
    }
}
