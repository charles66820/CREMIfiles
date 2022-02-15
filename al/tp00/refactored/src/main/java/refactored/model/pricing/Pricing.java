package refactored.model.pricing;


public interface Pricing extends Cloneable {
    double getPrice(double daysRented);
    int frequentRenterPoints(double daysRented);
    Pricing clone();
}
