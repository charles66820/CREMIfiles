package com.cgoedefroit.tdDp;

import com.cgoedefroit.tdDp.Soldier.Knight;
import com.cgoedefroit.tdDp.Soldier.Infantry;
import com.cgoedefroit.tdDp.Soldier.Soldier;
import com.cgoedefroit.tdDp.SoldierUtile.ShieldDecorator;
import com.cgoedefroit.tdDp.SoldierUtile.SoldierDecorator;
import com.cgoedefroit.tdDp.SoldierUtile.SoldierProxy;
import com.cgoedefroit.tdDp.SoldierUtile.SwordDecorator;

public class Main {

    public static void main(String[] args) {
        decoratorTests(false);
        proxyTests(true);
    }

    private static int fight(Soldier a, Soldier e, boolean debug) {
        int ncoups = 0;
        boolean la = true;
        boolean le = true;
        while (la && le) {
            ncoups++;
            if (debug) System.out.println("coups " + ncoups);
            int es = e.strength();
            la = a.wardOff(es);
            if (debug) System.out.println(e.getName() + " inflige " + es + " au " + a.getName());
            if (debug) System.out.println(a.getName() + " a " + a.getLifePoints() + " pv");
            if (la) {
                int as = a.strength();
                le = e.wardOff(as);
                if (debug) System.out.println(a.getName() + " inflige " + as + " au " + e.getName());
                if (debug) System.out.println(e.getName() + " a " + e.getLifePoints() + " pv");
            }
        }
        return ncoups;
    }

    private static void doFight(Soldier a, Soldier e, boolean debug) {
        int ncoups = fight(a, e, debug);
        System.out.println("Mort du " + (a.isAlive() ? e.getName() : a.getName())
                + " en " + ncoups + " coups\n");
    }

    private static void decoratorTests(boolean debug) {
        System.out.println("==== Decorator tests ====");
        System.out.println("Cavalier avec epee vs fantassin avec epee :");
        Soldier k = new SwordDecorator(new Knight(100));
        Soldier i = new SwordDecorator(new Infantry(50));
        doFight(k, i, debug);

        System.out.println("Cavalier avec epee vs fantassin avec bouclier :");
        k = new SwordDecorator(new Knight(100));
        i = new ShieldDecorator(new Infantry(50));
        doFight(k, i, debug);

        System.out.println("Cavalier avec epee vs fantassin nu :");
        k = new SwordDecorator(new Knight(100));
        i = new SoldierDecorator(new Infantry(50));
        doFight(k, i, debug);

        System.out.println("Cavalier avec bouclier vs fantassin avec epee :");
        k = new ShieldDecorator(new Knight(100));
        i = new SwordDecorator(new Infantry(50));
        doFight(k, i, debug);

        System.out.println("Cavalier avec bouclier vs fantassin avec bouclier :");
        k = new ShieldDecorator(new Knight(100));
        i = new ShieldDecorator(new Infantry(50));
        doFight(k, i, debug);

        System.out.println("Cavalier avec bouclier vs fantassin nu :");
        k = new ShieldDecorator(new Knight(100));
        i = new SoldierDecorator(new Infantry(50));
        doFight(k, i, debug);

        System.out.println("Cavalier nu vs fantassin avec epee :");
        k = new SoldierDecorator(new Knight(100));
        i = new SwordDecorator(new Infantry(50));
        doFight(k, i, debug);

        System.out.println("Cavalier nu vs fantassin avec bouclier :");
        k = new SoldierDecorator(new Knight(100));
        i = new ShieldDecorator(new Infantry(50));
        doFight(k, i, debug);

        System.out.println("Cavalier nu vs fantassin nu :");
        k = new SoldierDecorator(new Knight(100));
        i = new SoldierDecorator(new Infantry(50));
        doFight(k, i, debug);

    }

    private static void proxyTests(boolean debug) {
        System.out.println("==== Proxy tests ====");
        System.out.println("Cavalier vs fantassin :");

        SoldierProxy<Knight> k = new SoldierProxy<>(Knight.class, 100);
        if (k.addSword()) {
            if (debug) System.out.println("epee ajouter!");
        } else if (debug) System.out.println("le soldat a deja une epee!");
        if (k.addSword()) {
            if (debug) System.out.println("epee ajouter!");
        } else if (debug) System.out.println("le soldat a deja une epee!");
        if (k.addShield()) {
            if (debug) System.out.println("bouclier ajouter!");
        } else if (debug) System.out.println("le soldat a deja une bouclier!");

        SoldierProxy<Infantry> i = new SoldierProxy<>(Infantry.class, 50);
        if (i.addShield()) {
            if (debug) System.out.println("bouclier ajouter!");
        } else if (debug) System.out.println("le soldat a deja une bouclier!");

        doFight(k, i, debug);
    }

}