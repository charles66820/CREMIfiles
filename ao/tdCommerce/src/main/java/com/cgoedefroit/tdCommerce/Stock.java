package com.cgoedefroit.tdCommerce;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

public class Stock {
    private String name;
    private String address;
    private Set<Product> products;

    public Stock(String name, String address) {
        this.name = name;
        this.address = address;
        this.products = new HashSet<>();
    }

    public boolean addProduct(String name, int quantity) {
        return products.add(new Product(name, quantity));
    }

    public void printProductQuantity() {
        for (Product p : products) {
            System.out.println(p);
        }
    }

    public String getName() {
        return name;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Stock stock = (Stock) o;
        return name.equals(stock.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name);
    }
}
