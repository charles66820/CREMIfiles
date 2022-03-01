package model;

import model.pricing.ChildrenPricing;
import model.pricing.NewReleasePricing;
import model.pricing.RegularPricing;
import oldVersion.model.Customer;
import oldVersion.model.Movie;
import oldVersion.model.Rental;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class CustomerHtmlTest {
    private Customer oldCustomer;
    private model.Customer customer;

    @Before
    public void setUp() throws Exception {
        oldCustomer = new Customer("Toto");
        oldCustomer.addRental(new Rental(new Movie("Rogue One", Movie.NEW_RELEASE), 5));
        oldCustomer.addRental(new Rental(new Movie("Reine des neiges", Movie.CHILDRENS), 7));
        oldCustomer.addRental(new Rental(new Movie("Star Wars III", Movie.REGULAR), 4));

        customer = new model.Customer("Toto");
        customer.addRental(new model.Rental(new model.Movie("Rogue One", new NewReleasePricing()), 5));
        customer.addRental(new model.Rental(new model.Movie("Reine des neiges", new ChildrenPricing()), 7));
        customer.addRental(new model.Rental(new model.Movie("Star Wars III", new RegularPricing()), 4));
    }

    @Test
    public void getName() {
        assertEquals(oldCustomer.getName(), customer.getName());
    }

    @Test
    public void statement() {
        assertEquals(oldCustomer.statementHTML(), customer.statementHTML());
    }
}
