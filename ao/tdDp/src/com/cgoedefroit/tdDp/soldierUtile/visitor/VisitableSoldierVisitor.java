package com.cgoedefroit.tdDp.soldierUtile.visitor;

import com.cgoedefroit.tdDp.soldier.Infantry;
import com.cgoedefroit.tdDp.soldier.Knight;
import com.cgoedefroit.tdDp.soldier.Soldier;
import com.cgoedefroit.tdDp.soldierUtile.SoldierComposite;
import com.cgoedefroit.tdDp.soldierUtile.SoldierProxy;
import com.cgoedefroit.tdDp.soldierUtile.decorator.DaggerDecorator;
import com.cgoedefroit.tdDp.soldierUtile.decorator.ShieldDecorator;
import com.cgoedefroit.tdDp.soldierUtile.decorator.SwordDecorator;

public interface VisitableSoldierVisitor {
    void visit(Infantry infantry);
    void visit(Knight knight);
    void visit(DaggerDecorator dagger);
    void visit(ShieldDecorator shield);
    void visit(SwordDecorator sword);
    void visit(SoldierComposite army);
    void visit(SoldierProxy<Soldier> proxy);
}
