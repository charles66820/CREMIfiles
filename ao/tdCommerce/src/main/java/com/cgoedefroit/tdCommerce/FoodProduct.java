package com.cgoedefroit.tdCommerce;

import java.time.LocalDate;

public class FoodProduct extends ProductExpirable {
    public FoodProduct(String name, int quantity, LocalDate date) {
        super(name, quantity, date);
    }
}
