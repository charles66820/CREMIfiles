package com.cgoedefroit.tdDp.SoldatUtile;

import com.cgoedefroit.tdDp.Soldat.Soldat;

public class SoldatDecorator extends AbstractSoldatDecorator {
    public SoldatDecorator(Soldat soldat) {
        super(soldat);
    }

    public int hit() {
        return soldat.force(); // * weapon hit damage
    }
}
