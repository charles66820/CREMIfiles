package com.cgoedefroit.tdDp.soldierUtile;

import com.cgoedefroit.tdDp.soldier.Soldier;

import java.util.ArrayList;
import java.util.List;

public class SoldierComposite implements Soldier {
    private final List<Soldier> childSoldier = new ArrayList<>();

    private final String name;

    public SoldierComposite(String name) {
        this.name = name;
    }

    public void add(Soldier graphic) {
        childSoldier.add(graphic);
    }

    public void remove(Soldier graphic) {
        childSoldier.remove(graphic);
    }

    private int countAlive() {
        return childSoldier.stream().reduce(0, (acc, s) -> s.isAlive() ? acc + 1 : acc, Integer::sum);
    }

    @Override
    public int strength() {
        int strength = 0;
        for (Soldier s : childSoldier)
            if (s.isAlive())
                strength += s.strength();
        return strength;
    }

    @Override
    public boolean wardOff(int strength) {
        if (!isAlive()) return false;
        int nbAlive = countAlive();
        int sStrength = strength / nbAlive;
        int remainder = strength % nbAlive;
        boolean hasHit = false;
        int i = 0;
        for (Soldier s : childSoldier)
            if (s.isAlive()) {
                if (i == nbAlive - 1) sStrength += remainder;
                if (s.wardOff(sStrength)) hasHit = true;
                i++;
            }
        return hasHit;
    }

    @Override
    public boolean isAlive() {
        return childSoldier.stream().anyMatch(Soldier::isAlive);
    }

    @Override
    public int getLifePoints() {
        int lifePoints = 0;
        for (Soldier s : childSoldier)
            if (s.isAlive()) lifePoints += s.getLifePoints();
        return lifePoints;
    }

    @Override
    public String getName() {
        return name;
    }
}
