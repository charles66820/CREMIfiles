package refactored.model;

public class Rental {
    private final Movie _movie;
    private final int _daysRented;

    public Rental(Movie movie, int daysRented) {
        _movie = movie;
        _daysRented = daysRented;
    }

    public int getDaysRented() {
        return _daysRented;
    }

    public Movie getMovie() {
        return _movie;
    }

    public double amount() {
        double thisAmount = 0;

        switch (_movie.getPriceCode()) {
            case Movie.REGULAR:
                thisAmount += getPrice(2, 1.5, 2, _daysRented);
                break;
            case Movie.NEW_RELEASE:
                thisAmount += getPrice(_daysRented * 3, 0, 0, _daysRented);
                break;
            case Movie.CHILDRENS:
                thisAmount += getPrice(1.5, 1.5, 3, _daysRented);
                break;
        }

        return thisAmount;
    }

    private double getPrice(double rentePrice, double renteExceedPrice, double AllowRenteDays, double daysRented) {
        double price = rentePrice;
        if (daysRented > AllowRenteDays)
            price += (daysRented - AllowRenteDays) * renteExceedPrice;
        return price;
    }

    public int frequentRenterPoints() {
        int frequentRenterPoints = 1;
        if ((_movie.getPriceCode() == Movie.NEW_RELEASE) && (_daysRented > 1))
            frequentRenterPoints++;
        return frequentRenterPoints;
    }
}
