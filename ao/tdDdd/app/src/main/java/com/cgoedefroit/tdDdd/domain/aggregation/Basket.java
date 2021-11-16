package com.cgoedefroit.tdDdd.domain.aggregation;
import com.cgoedefroit.tdDdd.exception.IntValidationException;
import com.cgoedefroit.tdDdd.domain.valueObject.CommandLine;
import com.cgoedefroit.tdDdd.domain.valueObject.Product;

import java.util.HashSet;
import java.util.Set;

public class Basket implements Aggregation {
    private final int id;
    private final Set<CommandLine> commandLines = new HashSet<>();
    private int amount;
    private boolean isValidate;

    public Basket(int id) {
        this.id = id;
    }

    public boolean addProduct(Product product, Integer quantity) throws IntValidationException {
        if (isValidate) return false;
        if (!commandLines.add(new CommandLine(product, quantity))) return false;
        amount = commandLines.stream().reduce(0, (acc, commandLine) -> acc += commandLine.getAmount(), Integer::sum);
        return true;
    }

    public boolean removeProduct(Product product) throws IntValidationException {
        if (isValidate) return false;
        if (!commandLines.remove(new CommandLine(product, 1))) return false;
        amount = commandLines.stream().reduce(0, (acc, commandLine) -> acc += commandLine.getAmount(), Integer::sum);
        return true;
    }

    public int getId() {
        return id;
    }

    public Set<CommandLine> getCommandLines() {
        return new HashSet<>(commandLines);
    }

    public int getAmount() {
        return amount;
    }

    public boolean isValidate() {
        return isValidate;
    }

    public void toggleValidate() {
        isValidate = !isValidate;
    }
}