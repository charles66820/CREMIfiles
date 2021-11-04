package com.cgoedefroit.tdDp.SoldierUtile;

import com.cgoedefroit.tdDp.Soldier.Soldier;

public class ShieldDecorator extends AbstractSoldierDecorator {
    private static final int SHIELD_STRENGTH = 7;
    private static final int SHIELD_RESISTANCE = 12;

    public ShieldDecorator(Soldier soldier) {
        super(soldier);
    }

    @Override
    public int strength() {
        return soldier.strength() * SHIELD_STRENGTH;
    }

    @Override
    public boolean wardOff(int strenght) {
        return super.wardOff(strenght <= SHIELD_RESISTANCE ? 0 : strenght - SHIELD_RESISTANCE);
    }
}
