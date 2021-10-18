package com.cgoedefroit.tdDp;

import com.cgoedefroit.tdDp.Soldier.Knight;
import com.cgoedefroit.tdDp.Soldier.Infantry;
import com.cgoedefroit.tdDp.Soldier.Soldier;
import com.cgoedefroit.tdDp.SoldierUtile.ShieldDecorator;
import com.cgoedefroit.tdDp.SoldierUtile.SoldierDecorator;

public class Main {

    public static void main(String[] args) {
        Soldier c = new SoldierDecorator(new Knight(100));
        Soldier f = new ShieldDecorator(new Infantry(50));
        int ncoups = 0;
        boolean vc = true;
        boolean vf = true;

        for (; (vf = f.wardOff(c.strength())) && (vc = c.wardOff(f.strength())); ncoups++)
            ;

        System.out.println("Mort du " + (vf ? "cavalier" : "fantassin")
                + " en " + ncoups + " coups");
    }
}