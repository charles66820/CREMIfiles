package model.pricing;

public class StandardPricing implements Pricing, Cloneable {
    private double rentePrice;
    private double renteExceedPrice;
    private double allowRenteDays;

    public StandardPricing(double rentePrice, double renteExceedPrice, double allowRenteDays) {
        this.rentePrice = rentePrice;
        this.renteExceedPrice = renteExceedPrice;
        this.allowRenteDays = allowRenteDays;
    }

    public double getPrice(double daysRented) {
        double price = rentePrice;
        if (daysRented > allowRenteDays)
            price += (daysRented - allowRenteDays) * renteExceedPrice;
        return price;
    }

    public int frequentRenterPoints(double daysRented) {
        return 1; // frequentRenterPoints
    }

    @Override
    public StandardPricing clone() {
        try {
            StandardPricing clone = (StandardPricing) super.clone();
            clone.rentePrice = this.rentePrice;
            clone.renteExceedPrice = this.renteExceedPrice;
            clone.allowRenteDays = this.allowRenteDays;
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new AssertionError();
        }
    }
}
