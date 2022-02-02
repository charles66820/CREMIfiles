package com.cgoedefroit.tdDp.soldier;

import com.cgoedefroit.tdDp.soldierUtile.visitor.VisitableSoldier;

public interface Soldier extends VisitableSoldier {
    public int strength();

    public boolean wardOff(int strength);

    public boolean isAlive();

    public int getLifePoints();

    public String getName();
}