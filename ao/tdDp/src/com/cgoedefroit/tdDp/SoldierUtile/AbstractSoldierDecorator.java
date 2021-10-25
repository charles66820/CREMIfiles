package com.cgoedefroit.tdDp.SoldierUtile;

import com.cgoedefroit.tdDp.Soldier.Soldier;

public abstract class AbstractSoldierDecorator implements Soldier {
    protected Soldier soldier;
    public AbstractSoldierDecorator(Soldier soldier) {
        this.soldier = soldier;
    }

    public int strength() {
        return soldier.strength();
    }

    public boolean wardOff(int strength) {
        return soldier.wardOff(strength);
    }

    public boolean isAlive() {
        return soldier.isAlive();
    }

    public String getName() {
        return soldier.getName();
    }

    public int getLifePoints() {
        return soldier.getLifePoints();
    }
}
