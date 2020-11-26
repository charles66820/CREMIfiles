package com.cgoedefroit.td7;

public class Point2D implements Comparable {
    private int x, y;
    private final String name;

    public Point2D(int x, int y) {
        this(x, y, "undefine");
    }

    public Point2D(int x, int y, String name) {
        this.x = x;
        this.y = y;
        this.name = name;
    }

    public Point2D(Point2D p) {
        this(p.x, p.y, p.name);
    }

    // Get / set
    public int getX() {
        return this.x;
    }

    public void setX(int v) {
        this.x = v;
    }

    public int getY() {
        return this.y;
    }

    public void setY(int v) {
        this.y = v;
    }

    // Methods
    public void translate(int dx, int dy) {
        this.x += dx;
        this.y += dy;
    }

    public void translate(int d) {
        this.x += d;
        this.y += d;
    }

    public boolean inside(Point2D p) {
        return this.equals(p);
    }

    public int distance(Point2D p) {
        return (int) Math.sqrt(Math.pow(p.x - this.x, 2) + Math.pow(p.y - this.y, 2));
    }

    @Override
    public String toString() {
        return "Point2D ( " + this.name + ", " + x + ", " + y + " )@" + this.hashCode();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || this.getClass() != o.getClass()) return false;
        Point2D point2D = (Point2D) o;
        return Double.compare(point2D.x, this.x) == 0 &&
                Double.compare(point2D.y, this.y) == 0;
    }

    @Override
    public int compareTo(Object o) {
        if (o instanceof Point2D) {
            Point2D p = (Point2D) o;
            int r = Integer.compare(this.getX(), p.getX());
            return (r != 0)? r : Integer.compare(this.getY(), p.getY());
        }
        return -1;
    }
}
