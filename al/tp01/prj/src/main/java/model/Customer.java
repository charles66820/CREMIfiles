package model;

import utile.StringStatementBuilder;

import java.util.*;

public class Customer {
    private final String _name;
    private final Vector<Rental> _rentals = new Vector<>();

    public Customer(String name) {
        _name = name;
    }

    public void addRental(Rental rental) {
        _rentals.addElement(rental);
    }

    public String getName() {
        return _name;
    }

    @Deprecated
    public String statement() {
        StringStatementBuilder statementBuilder = new StringStatementBuilder();
        statementBuilder.setName(_name);
        statementBuilder.setRental(new LinkedList<>(_rentals));
        return statementBuilder.getResult();
    }
}
 