package com.cgoedefroit.tdDdd.domain.valueObject;

import com.cgoedefroit.tdDdd.exception.IntValidationException;
import com.cgoedefroit.tdDdd.exception.StringValidationException;
import com.cgoedefroit.tdDdd.exception.ValidationException;

import java.util.Objects;

public class Product implements ValueObject {
    private final String referance;
    private final String name;
    private final String description;
    private final int price;

    public Product(String referance, String name, String description, int price) throws ValidationException {
        if (referance == null || referance.length() == 0 || referance.length() > 20 || referance.matches("^[a-zA-Z0-9_]*$"))
            throw new StringValidationException("Referance length need to be between 1 and 20 and referance need to be alphanumeric", 1, 20, true);
        if (name == null || name.length() == 0 || name.length() > 20)
            throw new StringValidationException("Name length need to be between 1 and 20", 1, 20, false);
        if (description != null && description.length() > 220)
            throw new StringValidationException("Description length can't be grater then 220", 0, 220, false);
        if (price < 0)
            throw new IntValidationException("Prive can't be lower then 0", 0);

        this.referance = referance;
        this.name = name;
        this.description = description;
        this.price = price;
    }

    public String getReferance() {
        return referance;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public int getPrice() {
        return price;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Product product = (Product) o;
        return price == product.price && referance.equals(product.referance) && name.equals(product.name) && Objects.equals(description, product.description);
    }

    @Override
    public int hashCode() {
        return Objects.hash(referance, name, description, price);
    }
}
