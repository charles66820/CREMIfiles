package model;

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

    public String statement() {
        double totalAmount = 0;
        int frequentRenterPoints = 0;

        StringBuilder result = new StringBuilder("Rental Record for " + getName() + "\n");

        for (Rental rental : new LinkedList<>(_rentals)) {
            double thisAmount = rental.amount();
            frequentRenterPoints += rental.frequentRenterPoints();

            result.append("\t").append(rental.getMovie().getTitle());
            result.append("\t").append(thisAmount).append(" \n");
            totalAmount += thisAmount;
        }
        result.append("Amount owned is ").append(totalAmount).append("\n");
        result.append("You earned ").append(frequentRenterPoints).append(" frequent renter points");
        return result.toString();

    }
}
 