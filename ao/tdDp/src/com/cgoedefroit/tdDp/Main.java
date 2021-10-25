package com.cgoedefroit.tdDp;

import com.cgoedefroit.tdDp.Soldier.Knight;
import com.cgoedefroit.tdDp.Soldier.Infantry;
import com.cgoedefroit.tdDp.Soldier.Soldier;
import com.cgoedefroit.tdDp.SoldierUtile.ShieldDecorator;
import com.cgoedefroit.tdDp.SoldierUtile.SoldierDecorator;
import com.cgoedefroit.tdDp.SoldierUtile.SwordDecorator;

public class Main {

    public static void main(String[] args) {
        decoratorTests();
    }

    private static int fight(Soldier a, Soldier e) {
        int ncoups = 0;
        boolean la = true;
        boolean le = true;
        while (la && le) {
            ncoups++;
            System.out.println("coups " + ncoups);
            int es = e.strength();
            la = a.wardOff(es);
            System.out.println(e.getName() + " inflige " + es + " au " + a.getName());
            System.out.println(a.getName() + " a " + a.getLifePoints() + " pv");
            if (la) {
                int as = a.strength();
                le = e.wardOff(as);
                System.out.println(a.getName() + " inflige " + as + " au " + e.getName());
                System.out.println(e.getName() + " a " + e.getLifePoints() + " pv");
            }
        }
        return ncoups;
    }

    private static void doFight(Soldier a, Soldier e) {
        int ncoups = fight(a, e);
        System.out.println("Mort du " + (a.isAlive() ? e.getName() : a.getName())
                + " en " + ncoups + " coups\n");
    }

    private static void decoratorTests() {
        System.out.println("Cavalier avec epee vs fantassin avec epee :");
        Soldier k = new SwordDecorator(new Knight(100));
        Soldier i = new SwordDecorator(new Infantry(50));
        doFight(k, i);

        System.out.println("Cavalier avec epee vs fantassin avec bouclier :");
        k = new SwordDecorator(new Knight(100));
        i = new ShieldDecorator(new Infantry(50));
        doFight(k, i);

        System.out.println("Cavalier avec epee vs fantassin nu :");
        k = new SwordDecorator(new Knight(100));
        i = new SoldierDecorator(new Infantry(50));
        doFight(k, i);

        System.out.println("Cavalier avec bouclier vs fantassin avec epee :");
        k = new ShieldDecorator(new Knight(100));
        i = new SwordDecorator(new Infantry(50));
        doFight(k, i);

        System.out.println("Cavalier avec bouclier vs fantassin avec bouclier :");
        k = new ShieldDecorator(new Knight(100));
        i = new ShieldDecorator(new Infantry(50));
        doFight(k, i);

        System.out.println("Cavalier avec bouclier vs fantassin nu :");
        k = new ShieldDecorator(new Knight(100));
        i = new SoldierDecorator(new Infantry(50));
        doFight(k, i);

        System.out.println("Cavalier nu vs fantassin avec epee :");
        k = new SoldierDecorator(new Knight(100));
        i = new SwordDecorator(new Infantry(50));
        doFight(k, i);

        System.out.println("Cavalier nu vs fantassin avec bouclier :");
        k = new SoldierDecorator(new Knight(100));
        i = new ShieldDecorator(new Infantry(50));
        doFight(k, i);

        System.out.println("Cavalier nu vs fantassin nu :");
        k = new SoldierDecorator(new Knight(100));
        i = new SoldierDecorator(new Infantry(50));
        doFight(k, i);
    }
}