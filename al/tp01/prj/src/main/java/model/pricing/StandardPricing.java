package model.pricing;

import java.util.Objects;

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

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        StandardPricing that = (StandardPricing) o;
        return Double.compare(that.rentePrice, rentePrice) == 0 && Double.compare(that.renteExceedPrice, renteExceedPrice) == 0 && Double.compare(that.allowRenteDays, allowRenteDays) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(rentePrice, renteExceedPrice, allowRenteDays);
    }
}
