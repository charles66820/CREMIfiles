package com.cgoedefroit.tdDp.Soldier;

public class Knight extends AbstraitSoldier {
    private static final int KNIGHT_STRENGTH = 2;

    public Knight(int lifePoints) {
        super(lifePoints);
    }

    public int strength() {
        return KNIGHT_STRENGTH;
    }
}