package com.cgoedefroit.tdDp.soldierUtile.decorator;

import com.cgoedefroit.tdDp.soldier.Soldier;

public abstract class AbstractSoldierDecorator implements Soldier {
    protected Soldier soldier;
    private int durability;

    public AbstractSoldierDecorator(Soldier soldier, int durability) {
        this.soldier = soldier;
        this.durability = durability;
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

    public int getDurability() {
        return durability;
    }

    protected final void degradation(int degradation) {
        durability = Math.max(durability - degradation, 0);
    }

    public Soldier getSoldier() {
        return soldier;
    }
}
