package model;

import org.junit.Before;
import org.junit.Test;
import refactoring.model.Customers;
import refactoring.model.Movie;
import refactoring.model.Rental;

import static org.junit.Assert.*;

public class CustomersTest {
    private Customers oldCustomer;
    private model.Customers customer;

    @Before
    public void setUp() throws Exception {
        oldCustomer = new Customers("Toto");
        oldCustomer.addRental(new Rental(new Movie("Rogue One", Movie.NEW_RELEASE), 5));
        oldCustomer.addRental(new Rental(new Movie("Reine des neiges", Movie.CHILDRENS), 7));
        oldCustomer.addRental(new Rental(new Movie("Star Wars III", Movie.REGULAR), 4));

        customer = new model.Customers("Toto");
        customer.addRental(new model.Rental(new model.Movie("Rogue One", model.Movie.NEW_RELEASE), 5));
        customer.addRental(new model.Rental(new model.Movie("Reine des neiges", model.Movie.CHILDRENS), 7));
        customer.addRental(new model.Rental(new model.Movie("Star Wars III", model.Movie.REGULAR), 4));
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
