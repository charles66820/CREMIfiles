package com.cgoedefroit.tdDp;

import com.cgoedefroit.tdDp.Soldier.Knight;
import com.cgoedefroit.tdDp.Soldier.Infantry;
import com.cgoedefroit.tdDp.Soldier.Soldier;
import com.cgoedefroit.tdDp.SoldierUtile.ShieldDecorator;
import com.cgoedefroit.tdDp.SoldierUtile.SoldierComposite;
import com.cgoedefroit.tdDp.SoldierUtile.SoldierProxy;
import com.cgoedefroit.tdDp.SoldierUtile.SwordDecorator;

public class Main {

    public static void main(String[] args) {
        decoratorTests(false);
        proxyTests(true);
        compositeTests(true);
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
        i = new Infantry(50);
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
        i = new Infantry(50);
        doFight(k, i, debug);

        System.out.println("Cavalier nu vs fantassin avec epee :");
        k = new Knight(100);
        i = new SwordDecorator(new Infantry(50));
        doFight(k, i, debug);

        System.out.println("Cavalier nu vs fantassin avec bouclier :");
        k = new Knight(100);
        i = new ShieldDecorator(new Infantry(50));
        doFight(k, i, debug);

        System.out.println("Cavalier nu vs fantassin nu :");
        k = new Knight(100);
        i = new Infantry(50);
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
        if (i.addDagger()) {
            if (debug) System.out.println("dague ajouter!");
        } else if (debug) System.out.println("le soldat a deja une dague!");
        if (i.addShield()) {
            if (debug) System.out.println("bouclier ajouter!");
        } else if (debug) System.out.println("le soldat a deja une bouclier!");

        doFight(k, i, debug);
    }

    private static void compositeTests(boolean debug) {
        System.out.println("==== Compsite tests ====");

        System.out.println("Test d'une armer avec 4 soldat (un chevalier avec 20 pv, un fantassin avec 10 pv et 2 soldat morts :");
        SoldierComposite armay1 = new SoldierComposite("Armay1");
        armay1.add(new Knight(0));
        armay1.add(new Knight(30));
        armay1.add(new Infantry(10));
        armay1.add(new Infantry(-1));
        System.out.println("L'armer a " + armay1.getLifePoints() + " pv et " + (armay1.isAlive() ? "est" : "n'est pas") + " en vie!");
        System.out.println("On inflige 30 dommage a l'armer");
        System.out.println(armay1.wardOff(30));
        System.out.println("L'armer a " + armay1.getLifePoints() + " pv et " + (armay1.isAlive() ? "est" : "n'est pas") + " en vie!");
        System.out.println("On inflige 5 dommage a l'armer");
        System.out.println(armay1.wardOff(5));
        System.out.println("L'armer a " + armay1.getLifePoints() + " pv et " + (armay1.isAlive() ? "est" : "n'est pas") + " en vie!");

        System.out.println("\nTest de combat d'armers :");
        SoldierComposite subArmay = new SoldierComposite("Subarmay");
        subArmay.add(new Infantry(5));
        subArmay.add(new Infantry(5));
        SoldierComposite armay2 = new SoldierComposite("Armay2");
        armay2.add(subArmay);
        armay2.add(new Knight(5));
        armay2.add(new Infantry(10));
        armay2.add(new Infantry(5));

        fight(armay1, armay2, debug);
    }

}