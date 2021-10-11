package com.cgoedefroit.tdCommerce;

import java.time.LocalDate;
import java.time.temporal.ChronoUnit;

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
        return (int) ChronoUnit.DAYS.between(LocalDate.now(), dllc);
    }

    public boolean expired() {
        return dllc.isBefore(LocalDate.now());
    }

    public boolean salable() {
        return this.expiredIn() >= 3;
    }

    @Override
    public String toString() {
        return super.toString() + ", expire in " + expiredIn() + (salable()? " is salable" : " is not salable");
    }
}
