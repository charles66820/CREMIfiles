package com.cgoedefroit.tdDp.SoldierUtile;

import com.cgoedefroit.tdDp.Soldier.Soldier;

public class SwordDecorator extends SoldierDecorator {
    private static final int SWORD_STRENGTH = 16;
    private static final int SWORD_RESISTANCE = 2;

    public SwordDecorator(Soldier soldier) {
        super(soldier);
    }

    @Override
    public int strength() {
        return soldier.strength() * SWORD_STRENGTH;
    }

    @Override
    public boolean wardOff(int strenght) {
        return super.wardOff(strenght - SWORD_RESISTANCE);
    }
}
