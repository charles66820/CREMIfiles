package com.cgoedefroit.tdDp.Soldat;

abstract class SoldatAbstrait implements Soldat {
    protected int vie;

    SoldatAbstrait(int vie) {
        this.vie = vie;
    }

    public boolean parer(int force) {
        vie = (vie > force) ? vie - force : 0;
        return vie > 0;
    }
}