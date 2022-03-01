package model;

import utile.HtmlStatementBuilder;
import utile.StatementBuilder;
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

    public void completeBuilder(StatementBuilder statementBuilder) {
        statementBuilder.setName(_name);
        statementBuilder.setRental(new LinkedList<>(_rentals));
    }

    @Deprecated
    public String statement() {
        StringStatementBuilder statementBuilder = new StringStatementBuilder();
        statementBuilder.setName(_name);
        statementBuilder.setRental(new LinkedList<>(_rentals));
        return statementBuilder.getResult();
    }

    @Deprecated
    public String statementHTML() {
        HtmlStatementBuilder statementBuilder = new HtmlStatementBuilder();
        statementBuilder.setName(_name);
        statementBuilder.setRental(new LinkedList<>(_rentals));
        return statementBuilder.getResult();
    }
}
 