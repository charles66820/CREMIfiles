package com.cgoedefroit.tdDp.soldier;

abstract class AbstraitSoldier implements Soldier {
    protected int lifePoints;

    AbstraitSoldier(int lifePoints) {
        this.lifePoints = lifePoints;
    }

    public boolean wardOff(int strength) {
        lifePoints = (lifePoints > strength) ? lifePoints - strength : 0;
        return lifePoints > 0;
    }

    public boolean isAlive() {
        return lifePoints > 0;
    }

    public int getLifePoints() {
        return lifePoints;
    }
}