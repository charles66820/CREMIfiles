package com.cgoedefroit.tdDp.Soldat;

public class Sword extends weapon {
    @Override
    public int force() {
        return 8;
    }

    @Override
    public boolean parer(int force) {
        return true;//2
    }
}
