package com.cgoedefroit.tdDp.soldier;

public class Knight extends AbstraitSoldier {
    private static final int KNIGHT_STRENGTH = 2;

    public Knight(int lifePoints) {
        super(lifePoints);
    }

    public int strength() {
        return KNIGHT_STRENGTH;
    }

    public String getName() {
        return "cavalier";
    }
}