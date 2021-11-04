package com.cgoedefroit.tdDp.SoldierUtile;

import com.cgoedefroit.tdDp.Soldier.Soldier;

import java.util.ArrayList;
import java.util.List;

public class SoldierComposite implements Soldier {
    private final List<Soldier> childSoldier = new ArrayList<>();

    public void add(Soldier graphic) {
        childSoldier.add(graphic);
    }

    public void remove(Soldier graphic) {
        childSoldier.remove(graphic);
    }

    @Override
    public int strength() {
        int strength = 0;
        for (Soldier s : childSoldier)
            strength += s.strength();
        return strength;
    }

    @Override
    public boolean wardOff(int strength) {
        int sStrength = strength / childSoldier.size();
        int remainder = strength % childSoldier.size();
        int i = 0;
        boolean hasHit = false;
        for (Soldier s : childSoldier) {
            if (i == childSoldier.size() - 1) sStrength += remainder;
            hasHit = s.wardOff(sStrength);
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
            lifePoints += s.getLifePoints();
        return lifePoints;
    }

    @Override
    public String getName() {
        return "Armay";
    }
}
