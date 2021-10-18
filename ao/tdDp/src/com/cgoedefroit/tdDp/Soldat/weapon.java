package com.cgoedefroit.tdDp.Soldat;

public abstract class weapon implements Soldat {
    @Override
    public int force() {
        return 0;
    }

    @Override
    public boolean parer(int force) {
        return false;
    }
}
