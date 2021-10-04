package com.cgoedefroit.tdCommerce;

import java.time.LocalDate;

public interface ExpirationDate {
    public String getDllc();
    /**
     * Return the number of days the product expired
     *
     * @return an number of days
     */
    public int expiredIn();
    public boolean expired();
}
