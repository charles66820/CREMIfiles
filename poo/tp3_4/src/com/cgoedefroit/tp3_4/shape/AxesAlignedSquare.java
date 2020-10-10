package com.cgoedefroit.tp3_4.shape;

import com.cgoedefroit.tp3_4.shape.elementary.Point2D;

public class AxesAlignedSquare extends AxesAlignedRectangle {

    private double side;

    public AxesAlignedSquare(Point2D pos, double side) {
        this(pos, side, "undefine");
    }

    public AxesAlignedSquare(Point2D pos, double side, String name) {
        super(pos, side, side);
        this.side = side;
        this.name = name;
    }

    // Get / set
    public double getSide() {
        return side;
    }

    public void setSide(double side) {
        this.side = side;
    }

    // Methods
    @Override
    public String toString() {
        return "Square ( " + this.name + ", " + this.side + ", " + this.pos + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || this.getClass() != o.getClass()) return false;
        AxesAlignedSquare that = (AxesAlignedSquare) o;
        return this.side == that.side &&
                this.pos.equals(that.pos) &&
                this.r == that.r &&
                this.g == that.g &&
                this.b == that.b;
    }
}
