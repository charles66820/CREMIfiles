package model;

import model.pricing.Pricing;

import java.util.Objects;

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

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Rental rental = (Rental) o;
        return _daysRented == rental._daysRented && _movie.equals(rental._movie) && _pricing.equals(rental._pricing);
    }

    @Override
    public int hashCode() {
        return Objects.hash(_movie, _daysRented, _pricing);
    }
}
