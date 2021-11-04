package com.cgoedefroit.tdDp.SoldierUtile;

import com.cgoedefroit.tdDp.Soldier.Soldier;

public class DaggerDecorator extends AbstractSoldierDecorator {
    private static final int SWORD_STRENGTH = 12;
    private static final int SWORD_RESISTANCE = 0;

    public DaggerDecorator(Soldier soldier) {
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
