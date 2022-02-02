package com.cgoedefroit.tdDp.soldierUtile.visitor;

import com.cgoedefroit.tdDp.soldier.Infantry;
import com.cgoedefroit.tdDp.soldier.Knight;
import com.cgoedefroit.tdDp.soldier.Soldier;
import com.cgoedefroit.tdDp.soldierUtile.SoldierComposite;
import com.cgoedefroit.tdDp.soldierUtile.SoldierProxy;
import com.cgoedefroit.tdDp.soldierUtile.decorator.DaggerDecorator;
import com.cgoedefroit.tdDp.soldierUtile.decorator.ShieldDecorator;
import com.cgoedefroit.tdDp.soldierUtile.decorator.SwordDecorator;

public class CountArmyMemberVisitor implements SoldierVisitor {
    private int infantryCounter;
    private int knightCounter;
    private boolean imFirst = true;

    public void visit(Infantry infantry) {
        printResult(() -> infantryCounter++);
    }

    public void visit(Knight knight) {
        printResult(() -> knightCounter++);
    }

    public void visit(DaggerDecorator dagger) {
        printResult(() -> dagger.getSoldier().accept(this));
    }

    public void visit(ShieldDecorator shield) {
        printResult(() -> shield.getSoldier().accept(this));
    }

    public void visit(SwordDecorator sword) {
        printResult(() -> sword.getSoldier().accept(this));
    }

    public void visit(SoldierComposite army) {
        printResult(() -> army.getChildSoldier().forEach(soldier -> soldier.accept(this)));
    }

    public void visit(SoldierProxy<Soldier> proxy) {
        printResult(() -> proxy.getSoldier().accept(this));
    }

    private void printResult(Callable f) {
        boolean printResult = imFirst;
        if (imFirst) imFirst = false;
        f.call();
        if (printResult)
            System.out.println("Il y a " + infantryCounter + " fantassins et " + knightCounter + " cavalier dans larmée.");
    }

    private interface Callable {
        void call();
    }
}
