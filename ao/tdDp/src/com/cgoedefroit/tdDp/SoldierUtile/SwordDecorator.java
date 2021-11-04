package com.cgoedefroit.tdDp.SoldierUtile;

import com.cgoedefroit.tdDp.Soldier.Soldier;

public class SwordDecorator extends AbstractSoldierDecorator {
    private static final int SWORD_STRENGTH = 16;
    private static final int SWORD_RESISTANCE = 2;
    private static final int SWORD_DURABILITY = 400;

    public SwordDecorator(Soldier soldier) {
        super(soldier, SWORD_DURABILITY);
    }

    @Override
    public int strength() {
        if (super.getDurability() > 0) {
            super.degradation(1);
            return super.strength() * SWORD_STRENGTH;
        } else return super.strength();
    }

    @Override
    public boolean wardOff(int strenght) {
        if (super.getDurability() > 0) {
            super.degradation(4);
            return super.wardOff(strenght <= SWORD_RESISTANCE ? 0 : strenght - SWORD_RESISTANCE);
        } else return super.wardOff(strenght);
    }
}
