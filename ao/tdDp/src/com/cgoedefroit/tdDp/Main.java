package com.cgoedefroit.tdDp;

import com.cgoedefroit.tdDp.Soldier.Knight;
import com.cgoedefroit.tdDp.Soldier.Infantry;
import com.cgoedefroit.tdDp.Soldier.Soldier;
import com.cgoedefroit.tdDp.SoldierUtile.ShieldDecorator;
import com.cgoedefroit.tdDp.SoldierUtile.SoldierDecorator;
import com.cgoedefroit.tdDp.SoldierUtile.SwordDecorator;

public class Main {

    public static void main(String[] args) {
        System.out.println("Cavalier avec epee vs fantassin avec epee :");
        Soldier k = new SwordDecorator(new Knight(100));
        Soldier i = new SwordDecorator(new Infantry(50));

        int ncoups = fight(k, i);
        System.out.println("Mort du " + (k.isAlive() ? "cavalier" : "fantassin")
                + " en " + ncoups + " coups");

        System.out.println("Cavalier avec epee vs fantassin avec bouclier :");
        k = new SwordDecorator(new Knight(100));
        i = new ShieldDecorator(new Infantry(50));
        ncoups = fight(k, i);
        System.out.println("Mort du " + (k.isAlive() ? "cavalier" : "fantassin")
                + " en " + ncoups + " coups");

        System.out.println("Cavalier avec epee vs fantassin nu :");
        k = new SwordDecorator(new Knight(100));
        i = new SoldierDecorator(new Infantry(50));
        ncoups = fight(k, i);
        System.out.println("Mort du " + (k.isAlive() ? "cavalier" : "fantassin")
                + " en " + ncoups + " coups");

        System.out.println("Cavalier avec bouclier vs fantassin avec epee :");
        k = new ShieldDecorator(new Knight(100));
        i = new SwordDecorator(new Infantry(50));
        ncoups = fight(k, i);
        System.out.println("Mort du " + (k.isAlive() ? "cavalier" : "fantassin")
                + " en " + ncoups + " coups");

        System.out.println("Cavalier avec bouclier vs fantassin avec bouclier :");
        k = new ShieldDecorator(new Knight(100));
        i = new ShieldDecorator(new Infantry(50));
        ncoups = fight(k, i);
        System.out.println("Mort du " + (k.isAlive() ? "cavalier" : "fantassin")
                + " en " + ncoups + " coups");

        System.out.println("Cavalier avec bouclier vs fantassin nu :");
        k = new ShieldDecorator(new Knight(100));
        i = new SoldierDecorator(new Infantry(50));
        ncoups = fight(k, i);
        System.out.println("Mort du " + (k.isAlive() ? "cavalier" : "fantassin")
                + " en " + ncoups + " coups");

        System.out.println("Cavalier nu vs fantassin avec epee :");
        k = new SoldierDecorator(new Knight(100));
        i = new SwordDecorator(new Infantry(50));
        ncoups = fight(k, i);
        System.out.println("Mort du " + (k.isAlive() ? "cavalier" : "fantassin")
                + " en " + ncoups + " coups");

        System.out.println("Cavalier nu vs fantassin avec bouclier :");
        k = new SoldierDecorator(new Knight(100));
        i = new ShieldDecorator(new Infantry(50));
        ncoups = fight(k, i);
        System.out.println("Mort du " + (k.isAlive() ? "cavalier" : "fantassin")
                + " en " + ncoups + " coups");

        System.out.println("Cavalier nu vs fantassin nu :");
        k = new SoldierDecorator(new Knight(100));
        i = new SoldierDecorator(new Infantry(50));
        ncoups = fight(k, i);
        System.out.println("Mort du " + (k.isAlive() ? "cavalier" : "fantassin")
                + " en " + ncoups + " coups");
    }

    private static int fight(Soldier a, Soldier e) {
        int ncoups = 0;
        boolean la = true;
        boolean le = true;
        while (la && le) {
            ncoups++;
            la = a.wardOff(e.strength());
            if (la) {
                le = e.wardOff(a.strength());
            }
        }
        return ncoups;
    }
}