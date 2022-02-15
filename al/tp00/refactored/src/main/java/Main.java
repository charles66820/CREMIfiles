import refactoring.model.Customers;
import refactoring.model.Movie;
import refactoring.model.Rental;

public class Main {
    public static void main(String[] args) {
        Customers c = new Customers("Toto");
        c.addRental(new Rental(new Movie("Rogue One", Movie.NEW_RELEASE), 5));
        c.addRental(new Rental(new Movie("Reine des neiges", Movie.CHILDRENS), 7));
        c.addRental(new Rental(new Movie("Star Wars III", Movie.REGULAR), 4));

        System.out.println("Films of " + c.getName());
        System.out.println(c.statement());
    }
}
