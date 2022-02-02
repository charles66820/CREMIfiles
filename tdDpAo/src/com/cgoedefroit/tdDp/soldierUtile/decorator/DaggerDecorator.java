package com.cgoedefroit.tdDp.soldierUtile.decorator;

import com.cgoedefroit.tdDp.soldier.Soldier;
import com.cgoedefroit.tdDp.soldierUtile.visitor.SoldierVisitor;

public class DaggerDecorator extends AbstractSoldierDecorator {
    private static final int DAGGER_STRENGTH = 12;
    private static final int DAGGER_RESISTANCE = 0;
    private static final int DAGGER_DURABILITY = 100;

    public DaggerDecorator(Soldier soldier) {
        super(soldier, DAGGER_DURABILITY);
    }

    @Override
    public int strength() {
        if (super.getDurability() > 0) {
            super.degradation(1);
            return super.strength() * DAGGER_STRENGTH;
        } else return super.strength();
    }

    @Override
    public boolean wardOff(int strength) {
        if (super.getDurability() > 0) {
            super.degradation(6);
            return super.wardOff(strength <= DAGGER_RESISTANCE ? 0 : strength - DAGGER_RESISTANCE);
        } else return super.wardOff(strength);
    }

    public void accept(SoldierVisitor visitor) {
        visitor.visit(this);
    }
}
