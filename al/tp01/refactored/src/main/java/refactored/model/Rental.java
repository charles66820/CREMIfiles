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
                thisAmount += 2; // pris du forfait
                if (_daysRented > 2) { // jour hore forfait
                    thisAmount += (_daysRented - 2) * 1.5; // pris du forfait * nb days
                }
                break;
            case Movie.NEW_RELEASE:
                thisAmount += _daysRented * 3; // pris du forfait * nb days
                break;
            case Movie.CHILDRENS:
                thisAmount += 1.5; // pris du forfait
                if (_daysRented > 3) // jour hore forfait
                    thisAmount += (_daysRented - 3) * 1.5; // pris du forfait * nb days
                break;
        }

        return thisAmount;
    }

    public int frequentRenterPoints() {
        int frequentRenterPoints = 1;
        if ((_movie.getPriceCode() == Movie.NEW_RELEASE) && (_daysRented > 1))
            frequentRenterPoints++;
        return frequentRenterPoints;
    }
}
