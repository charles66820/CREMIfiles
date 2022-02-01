package refactored.model;

public class Rental {
    private final Movie _movie;
    private final int _daysRented;
    private final Pricing _pricing;

    public Rental(Movie movie, int daysRented) {
        _movie = movie;
        _daysRented = daysRented;
        _pricing = movie.getPricing().clone();
    }

    public int getDaysRented() {
        return _daysRented;
    }

    public Movie getMovie() {
        return _movie;
    }

    public double amount() {
        return _pricing.getPrice(_daysRented);
    }

    public int frequentRenterPoints() {
        return _pricing.frequentRenterPoints(_daysRented);
    }
}
