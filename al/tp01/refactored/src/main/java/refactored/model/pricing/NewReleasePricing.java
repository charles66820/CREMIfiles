package refactored.model.pricing;

public class NewReleasePricing extends StandardPricing implements Cloneable {
    public NewReleasePricing() {
        super(3, 0, 0);
    }

    @Override
    public double getPrice(double daysRented) {
        return daysRented * rentePrice;
    }

    @Override
    public int frequentRenterPoints(double daysRented) {
        int frequentRenterPoints = 1;
        if (daysRented > 1)
            frequentRenterPoints++;
        return frequentRenterPoints;
    }

    @Override
    public NewReleasePricing clone() {
        return (NewReleasePricing) super.clone();
    }
}
