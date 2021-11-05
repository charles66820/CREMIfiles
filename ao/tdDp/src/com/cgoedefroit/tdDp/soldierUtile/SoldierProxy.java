package com.cgoedefroit.tdDp.soldierUtile;

import com.cgoedefroit.tdDp.soldier.Soldier;
import com.cgoedefroit.tdDp.soldierUtile.decorator.DaggerDecorator;
import com.cgoedefroit.tdDp.soldierUtile.decorator.ShieldDecorator;
import com.cgoedefroit.tdDp.soldierUtile.decorator.SwordDecorator;

import java.lang.reflect.InvocationTargetException;
import java.util.HashSet;
import java.util.Set;

public class SoldierProxy<T extends Soldier> implements Soldier {
    private final int lifePoints;
    private final Class<T> soldierClass;
    private final Set<String> equipments = new HashSet<>();
    private Soldier soldier;

    public SoldierProxy(Class<T> soldierClass, int lifePoints) {
        this.soldierClass = soldierClass;
        this.lifePoints = lifePoints;
    }

    public boolean addShield() {
        lazyLoad();
        boolean canBeAdd = equipments.add("SHIELD");
        if (canBeAdd) soldier = new ShieldDecorator(soldier);
        return canBeAdd;
    }

    public boolean addSword() {
        lazyLoad();
        boolean canBeAdd = equipments.add("SWORD");
        if (canBeAdd) soldier = new SwordDecorator(soldier);
        return canBeAdd;
    }

    public boolean addDagger() {
        lazyLoad();
        boolean canBeAdd = equipments.add("DAGGER");
        if (canBeAdd) soldier = new DaggerDecorator(soldier);
        return canBeAdd;
    }

    private Soldier lazyLoad() {
        if (soldier == null) {
            try {
                soldier = soldierClass.getDeclaredConstructor(int.class).newInstance(lifePoints);
            } catch (InstantiationException | IllegalAccessException | InvocationTargetException | NoSuchMethodException e) {
                e.printStackTrace();
            }
        }
        return soldier;
    }

    @Override
    public int strength() {
        return lazyLoad().strength();
    }

    @Override
    public boolean wardOff(int strength) {
        return lazyLoad().wardOff(strength);
    }

    @Override
    public boolean isAlive() {
        return lazyLoad().isAlive();
    }

    @Override
    public int getLifePoints() {
        return lazyLoad().getLifePoints();
    }

    @Override
    public String getName() {
        return lazyLoad().getName();
    }
}
