package com.cgoedefroit.tdCommerce;

import java.util.Objects;

public class Product {
    private String name;
    private int quantity;
    public final int reference;
    private static int counter = 0;

    public Product(String name, int quantity) {
        counter += 1;
        this.reference = counter;
        this.name = name;
        this.quantity = quantity;
    }

    public String getName() {
        return name;
    }

    public int getQuantity() {
        return quantity;
    }

    public void setQuantity(int quantity) {
        this.quantity = quantity;
    }

    @Override
    public String toString() {
        return '"' + name + "\" nb: " + quantity + ", ref: " + reference + '"';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Product product = (Product) o;
        return name.equals(product.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name);
    }
}
