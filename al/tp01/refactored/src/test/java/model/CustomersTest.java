package model;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import refactoring.model.Customers;
import refactoring.model.Movie;
import refactoring.model.Rental;

import static org.junit.jupiter.api.Assertions.*;

public class CustomersTest {
    private Customers oldCustomer;
    private refactored.model.Customers customer;

    @BeforeEach
    public void setUp() throws Exception {
        oldCustomer = new Customers("Toto");
        oldCustomer.addRental(new Rental(new Movie("Rogue One", Movie.NEW_RELEASE), 5));
        oldCustomer.addRental(new Rental(new Movie("Reine des neiges", Movie.CHILDRENS), 7));
        oldCustomer.addRental(new Rental(new Movie("Star Wars III", Movie.REGULAR), 4));

        customer = new refactored.model.Customers("Toto");
        customer.addRental(new refactored.model.Rental(new refactored.model.Movie("Rogue One", refactored.model.Movie.NEW_RELEASE), 5));
        customer.addRental(new refactored.model.Rental(new refactored.model.Movie("Reine des neiges", refactored.model.Movie.CHILDRENS), 7));
        customer.addRental(new refactored.model.Rental(new refactored.model.Movie("Star Wars III", refactored.model.Movie.REGULAR), 4));
    }

    @Test
    public void getName() {
        assertEquals(oldCustomer.getName(), customer.getName());
    }

    @Test
    public void statement() {
        assertEquals(oldCustomer.statement(), customer.statement());
    }
}
