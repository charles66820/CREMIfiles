package com.cgoedefroit.tp3.shape;

import com.cgoedefroit.tp3.shape.elementary.Point2D;

public class AxesAlignedSquare extends AxesAlignedRectangle {

    private int side;

    public AxesAlignedSquare(Point2D pos, int side) {
        this(pos, side, "undefine");
    }

    public AxesAlignedSquare(Point2D pos, int side, String name) {
        super(pos, side, side);
        this.side = side;
        this.name = name;
    }

    // Get / set
    public int getSide() {
        return side;
    }

    public void setSide(int side) {
        this.side = side;
    }

    // Methods
    @Override
    public String toString() {
        return "Square ( " + this.name + ", " + this.side + ", Point2D (" + this.pos.getX() + ", "  + this.pos.getY() + ")" + ")";
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
