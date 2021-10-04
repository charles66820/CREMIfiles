package com.cgoedefroit.tdCommerce;

import java.time.LocalDate;
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

    public boolean addSanitaryProduct(String name, int quantity) {
        return products.add(new SanitaryProduct(name, quantity));
    }

    public boolean addFoodProduct(String name, int quantity, LocalDate date) {
        return products.add(new FoodProduct(name, quantity, date));
    }

    public boolean hasProducts() {
        return products.size() != 0;
    }

    public void printProducts() {
        for (Product p : products) {
            System.out.println(" * " + p);
        }
    }

    private Product getProduct(String name) {
        for (Product product : products)
            if (Objects.equals(product.getName(), name))
                return product;
        return null;
    }

    public String getProductInfo(String name) {
        Product product = getProduct(name);
        if (product == null) return null;
        return product.toString();
    }

    public String getName() {
        return name;
    }

    boolean addProductQuantity(String name, int delta) throws Exception {
        Product product = getProduct(name);
        if (product == null) throw new Exception("Product not found");
        int newQuantity = product.getQuantity() + Math.abs(delta);
        if (newQuantity >= 0) {
            product.setQuantity(newQuantity);
            return true;
        }
        return false;
    }

    boolean subProductQuantity(String name, int delta) throws Exception {
        Product product = getProduct(name);
        if (product == null) throw new Exception("Product not found");
        int newQuantity = product.getQuantity() - Math.abs(delta);
        if (newQuantity >= 0) {
            product.setQuantity(newQuantity);
            return true;
        }
        return false;
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
