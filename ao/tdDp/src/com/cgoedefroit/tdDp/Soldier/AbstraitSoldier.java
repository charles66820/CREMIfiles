package com.cgoedefroit.tdDp.Soldier;

abstract class AbstraitSoldier implements Soldier {
    protected int lifePoints;

    AbstraitSoldier(int lifePoints) {
        this.lifePoints = lifePoints;
    }

    public boolean wardOff(int strength) {
        lifePoints = (lifePoints > strength) ? lifePoints - strength : 0;
        return lifePoints > 0;
    }
}