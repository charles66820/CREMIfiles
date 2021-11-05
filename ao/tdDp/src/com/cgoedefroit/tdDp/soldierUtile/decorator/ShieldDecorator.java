package com.cgoedefroit.tdDp.soldierUtile.decorator;

import com.cgoedefroit.tdDp.soldier.Soldier;

public class ShieldDecorator extends AbstractSoldierDecorator {
    private static final int SHIELD_STRENGTH = 7;
    private static final int SHIELD_RESISTANCE = 12;
    private static final int SHIELD_DURABILITY = 200;

    public ShieldDecorator(Soldier soldier) {
        super(soldier, SHIELD_DURABILITY);
    }

    @Override
    public int strength() {
        if (super.getDurability() > 0) {
            return super.strength() * SHIELD_STRENGTH;
        } else return super.strength();
    }

    @Override
    public boolean wardOff(int strength) {
        if (super.getDurability() > 0) {
            super.degradation(1);
            return super.wardOff(strength <= SHIELD_RESISTANCE ? 0 : strength - SHIELD_RESISTANCE);
        } else return super.wardOff(strength);
    }
}
