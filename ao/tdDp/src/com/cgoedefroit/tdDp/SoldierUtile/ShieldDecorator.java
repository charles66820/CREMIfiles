package com.cgoedefroit.tdDp.SoldierUtile;

import com.cgoedefroit.tdDp.Soldier.Soldier;

public class ShieldDecorator extends SoldierDecorator {
    private static final int SHIELD_STRENGTH = 4;
    private static final int SHIELD_RESISTANCE = 6;

    public ShieldDecorator(Soldier soldier) {
        super(soldier);
    }

    public int hit() {
        return soldier.strength() * SHIELD_STRENGTH;
    }

    @Override
    public boolean wardOff(int strenght) {
        return super.wardOff(strenght - SHIELD_RESISTANCE);
    }
}
