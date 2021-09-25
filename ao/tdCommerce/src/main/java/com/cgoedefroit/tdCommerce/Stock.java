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

    public boolean hasProducts() {
        return products.size() != 0;
    }

    public void printProducts() {
        for (Product p : products) {
            System.out.println(" * " + p);
        }
    }

    public Product getProduct(String name) {
        for (Product product : products)
            if (Objects.equals(product.getName(), name))
                return product;
        return null;
    }

    public String getName() {
        return name;
    }

    @Override
    public String toString() {
        return '"' + name + "\" at \"" + address + '"';
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
