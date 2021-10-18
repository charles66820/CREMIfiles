package com.cgoedefroit.tdDp.Soldat;

public class Shield extends weapon {
    @Override
    public int force() {
        return 4;
    }

    @Override
    public boolean parer(int force) {
        return true;//6
    }
}
