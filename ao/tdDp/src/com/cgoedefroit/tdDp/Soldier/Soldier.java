package com.cgoedefroit.tdDp.Soldier;

public interface Soldier {
    public int strength();

    public boolean wardOff(int strength);

    public boolean isAlive();

    public int getLifePoints();

    public String getName();
}