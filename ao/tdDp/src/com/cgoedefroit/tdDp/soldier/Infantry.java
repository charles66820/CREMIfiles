package com.cgoedefroit.tdDp.soldier;

public class Infantry extends AbstraitSoldier {
    private static final int INFANTRY_STRENGTH = 1;

    public Infantry(int lifePoints) {
        super(lifePoints);
    }

    public int strength() {
        return INFANTRY_STRENGTH;
    }

    public String getName() {
        return "fantassin";
    }
}