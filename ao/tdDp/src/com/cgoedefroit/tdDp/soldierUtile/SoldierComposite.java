package com.cgoedefroit.tdDp.soldierUtile;

import com.cgoedefroit.tdDp.soldier.Soldier;
import com.cgoedefroit.tdDp.soldierUtile.visitor.SoldierVisitor;

import java.util.ArrayList;
import java.util.List;

public class SoldierComposite implements Soldier {
    private final List<Soldier> childSoldier = new ArrayList<>();

    private final String name;

    public SoldierComposite(String name) {
        this.name = name;
    }

    public void add(Soldier soldier) {
        childSoldier.add(soldier);
    }

    public void remove(Soldier soldier) {
        childSoldier.remove(soldier);
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
        boolean isFirst = true;
        for (Soldier s : childSoldier)
            if (s.isAlive()) {
                if (s.wardOff(isFirst ? sStrength + remainder : sStrength)) hasHit = true;
                isFirst = false;
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

    public void accept(SoldierVisitor visitor) {
        visitor.visit(this);
    }

    public List<Soldier> getChildSoldier() {
        return childSoldier;
    }
}