package com.cgoedefroit.tdDp;

import com.cgoedefroit.tdDp.soldier.Knight;
import com.cgoedefroit.tdDp.soldier.Infantry;
import com.cgoedefroit.tdDp.soldier.Soldier;
import com.cgoedefroit.tdDp.soldierUtile.decorator.DaggerDecorator;
import com.cgoedefroit.tdDp.soldierUtile.decorator.ShieldDecorator;
import com.cgoedefroit.tdDp.soldierUtile.SoldierComposite;
import com.cgoedefroit.tdDp.soldierUtile.SoldierProxy;
import com.cgoedefroit.tdDp.soldierUtile.decorator.SwordDecorator;
import com.cgoedefroit.tdDp.soldierUtile.visitor.ShowArmyVisitor;

public class Main {

    public static void main(String[] args) {
        decoratorTests(false);
        proxyTests(true);
        compositeTests(true);
        visitorTests();
    }

    private static int fight(Soldier a, Soldier e, boolean debug) {
        int nCoups = 0;
        boolean la = true;
        boolean le = true;
        while (la && le) {
            nCoups++;
            if (debug) System.out.println("coups " + nCoups);
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
        return nCoups;
    }

    private static void doFight(Soldier a, Soldier e, boolean debug) {
        int nCoups = fight(a, e, debug);
        System.out.println("Mort du " + (a.isAlive() ? e.getName() : a.getName())
                + " en " + nCoups + " coups\n");
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
        System.out.println("==== Composite tests ====");

        System.out.println("Test d'une armer avec 4 soldat (un chevalier avec 20 pv, un fantassin avec 10 pv et 2 soldat morts :");
        SoldierComposite army1 = new SoldierComposite("Army1");
        army1.add(new Knight(0));
        army1.add(new Knight(30));
        army1.add(new Infantry(10));
        army1.add(new Infantry(-1));
        System.out.println("L'armer a " + army1.getLifePoints() + " pv et " + (army1.isAlive() ? "est" : "n'est pas") + " en vie!");
        System.out.println("On inflige 30 dommage a l'armer");
        System.out.println(army1.wardOff(30));
        System.out.println("L'armer a " + army1.getLifePoints() + " pv et " + (army1.isAlive() ? "est" : "n'est pas") + " en vie!");
        System.out.println("On inflige 5 dommage a l'armer");
        System.out.println(army1.wardOff(5));
        System.out.println("L'armer a " + army1.getLifePoints() + " pv et " + (army1.isAlive() ? "est" : "n'est pas") + " en vie!");

        System.out.println("\nTest de combat d'armées :");
        SoldierComposite subArmy = new SoldierComposite("Sub-army");
        subArmy.add(new Infantry(5));
        subArmy.add(new Infantry(5));
        SoldierComposite army2 = new SoldierComposite("Army2");
        army2.add(subArmy);
        army2.add(new Knight(5));
        army2.add(new Infantry(10));
        army2.add(new Infantry(5));

        fight(army1, army2, debug);
    }

    private static void visitorTests() {
        SoldierProxy<Infantry> specialSoldier = new SoldierProxy<>(Infantry.class, 10);
        specialSoldier.addSword();
        specialSoldier.addShield();

        SoldierComposite subArmy = new SoldierComposite("Sub-army");
        subArmy.add(new SwordDecorator(new Infantry(5)));
        subArmy.add(new SwordDecorator(new Infantry(5)));

        SoldierComposite army = new SoldierComposite("Army");
        army.add(new DaggerDecorator(new Knight(0)));
        army.add(new SwordDecorator(new Knight(30)));
        army.add(subArmy);
        army.add(specialSoldier);
        army.add(new Infantry(-1));

        System.out.println("==== Visitor tests ====");
        System.out.println("Afficher tous les membres d'une armée :\n");
        (new ShowArmyVisitor()).visit(army);
    }

}