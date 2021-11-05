package com.cgoedefroit.tdDp.soldier;

public class Infantry extends AbstraitSoldier {
    private static final int INFANTTY_STRENGTH = 1;

    public Infantry(int lifePoints) {
        super(lifePoints);
    }

    public int strength() {
        return INFANTTY_STRENGTH;
    }

    public String getName() {
        return "fantassin";
    }
}