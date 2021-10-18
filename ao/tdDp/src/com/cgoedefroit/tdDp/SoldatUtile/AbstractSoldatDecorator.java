package com.cgoedefroit.tdDp.SoldatUtile;

import com.cgoedefroit.tdDp.Soldat.Soldat;

public abstract class AbstractSoldatDecorator implements Soldat {
    protected Soldat soldat;
    public AbstractSoldatDecorator(Soldat soldat) {
        this.soldat = soldat;
    }

    public int force() {
        return soldat.force();
    }

    public boolean parer(int force) {
        return false;
    }
}
