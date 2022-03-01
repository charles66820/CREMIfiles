package utile;

import model.Rental;

import java.util.LinkedList;

public interface StatementBuilder {
    void setName(String name);
    void setRental(LinkedList<Rental> rentals);
}
