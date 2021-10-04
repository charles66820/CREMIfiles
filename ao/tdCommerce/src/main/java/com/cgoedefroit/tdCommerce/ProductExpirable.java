package com.cgoedefroit.tdCommerce;

import java.time.LocalDate;

public abstract class ProductExpirable extends Product implements ExpirationDate {
    private final LocalDate dllc;

    public ProductExpirable(String name, int quantity, LocalDate date) {
        super(name, quantity);
        this.dllc = LocalDate.from(date);
    }

    public String getDllc() {
        return dllc.toString();
    }

    /**
     * Return the number of days the product expired
     *
     * @return an number of days
     */
    public int expiredIn() {
        // TODO: date diff
        LocalDate.now();
        return 0;
    }

    public boolean expired() {
        return dllc.isBefore(LocalDate.now());
    }
}
