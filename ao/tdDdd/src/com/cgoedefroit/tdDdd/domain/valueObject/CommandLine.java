package com.cgoedefroit.tdDdd.domain.valueObject;

import com.cgoedefroit.tdDdd.exception.IntValidationException;

import java.util.Objects;

public class CommandLine implements ValueObject {
    private final Product product;
    private final int quantity;
    private final int amount;

    public CommandLine(Product product, int quantity) throws IntValidationException {
        this.product = product;
        if (quantity < 1)
            throw new IntValidationException("Quantity can't be lower then 1", 1);
        this.quantity = quantity;
        this.amount = product.getPrice() * quantity;
    }

    public Product getProduct() {
        return product;
    }

    public int getQuantity() {
        return quantity;
    }

    public int getAmount() {
        return amount;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        CommandLine that = (CommandLine) o;
        return product.equals(that.product);
    }

    @Override
    public int hashCode() {
        return Objects.hash(product);
    }
}
