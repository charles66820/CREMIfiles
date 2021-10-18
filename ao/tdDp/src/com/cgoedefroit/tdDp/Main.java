package com.cgoedefroit.tdDp;

import com.cgoedefroit.tdDp.Soldat.Cavalier;
import com.cgoedefroit.tdDp.Soldat.Fantassin;
import com.cgoedefroit.tdDp.Soldat.Soldat;

public class Main {

    public static void main(String[] args) {
        Soldat c = new Cavalier(100);
        Soldat f = new Fantassin(50);
        int ncoups = 0;
        boolean vc = true;
        boolean vf = true;

        for (; (vf = f.parer(c.force())) && (vc = c.parer(f.force())); ncoups++)
            ;

        System.out.println("Mort du " + (vf ? "cavalier" : "fantassin")
                + " en " + ncoups + " coups");
    }
}