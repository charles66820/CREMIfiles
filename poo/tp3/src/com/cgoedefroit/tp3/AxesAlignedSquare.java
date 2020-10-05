package com.cgoedefroit.tp3;

public class AxesAlignedSquare extends AxesAlignedRectangle {

    private int side;

    public AxesAlignedSquare(Point2D pos, int side) {
        super(pos, side, side);
        this.side = side;
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
        return "Square (" + this.side + ", Point2D (" + this.getPos().getX() + ", "  + this.getPos().getY() + ")" + ")";
    }
}
