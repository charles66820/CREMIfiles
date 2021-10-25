package com.cgoedefroit.tdDp.SoldierUtile;

import com.cgoedefroit.tdDp.Soldier.Soldier;

public class SoldierDecorator extends AbstractSoldierDecorator {
    private static final int SOLDIER_STRENGTH = 4;
    public SoldierDecorator(Soldier soldier) {
        super(soldier);
    }

    @Override
    public int strength() {
        return soldier.strength() * SOLDIER_STRENGTH;
    }
}
