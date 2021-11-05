package com.cgoedefroit.tdDp.soldierUtile.visitor;

import com.cgoedefroit.tdDp.soldier.Infantry;
import com.cgoedefroit.tdDp.soldier.Knight;
import com.cgoedefroit.tdDp.soldier.Soldier;
import com.cgoedefroit.tdDp.soldierUtile.SoldierComposite;
import com.cgoedefroit.tdDp.soldierUtile.SoldierProxy;
import com.cgoedefroit.tdDp.soldierUtile.decorator.DaggerDecorator;
import com.cgoedefroit.tdDp.soldierUtile.decorator.ShieldDecorator;
import com.cgoedefroit.tdDp.soldierUtile.decorator.SwordDecorator;

public class ShowArmyVisitor implements SoldierVisitor {
    public void visit(Infantry infantry) {
        System.out.println("Un " + infantry.getName());
    }

    public void visit(Knight knight) {
        System.out.println("Un " + knight.getName());
    }

    public void visit(DaggerDecorator dagger) {
        dagger.getSoldier().accept(this);
        System.out.println("avec une dague");
    }

    public void visit(ShieldDecorator shield) {
        shield.getSoldier().accept(this);
        System.out.println("avec un bouclier");
    }

    public void visit(SwordDecorator sword) {
        sword.getSoldier().accept(this);
        System.out.println("avec une épee");
    }

    public void visit(SoldierComposite army) {
        System.out.println("Les membres de l'armée " + army.getName() + " sont :");
        for (Soldier soldier : army.getChildSoldier())
            soldier.accept(this);
        System.out.println("L'armée " + army.getName() + " est visiter!");
    }

    public void visit(SoldierProxy<Soldier> proxy) {
        proxy.getSoldier().accept(this);
    }
}
