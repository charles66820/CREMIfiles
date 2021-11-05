package com.cgoedefroit.tdDp.soldier;

import com.cgoedefroit.tdDp.soldierUtile.visitor.VisitableSoldierVisitor;

public class Knight extends AbstraitSoldier {
    private static final int KNIGHT_STRENGTH = 2;

    public Knight(int lifePoints) {
        super(lifePoints);
    }

    public int strength() {
        return KNIGHT_STRENGTH;
    }

    public String getName() {
        return "cavalier";
    }

    public void accept(VisitableSoldierVisitor visitor) {
        visitor.visit(this);
    }
}