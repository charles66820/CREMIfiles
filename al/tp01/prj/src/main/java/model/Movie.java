package model;

import model.pricing.ChildrenPricing;
import model.pricing.NewReleasePricing;
import model.pricing.Pricing;
import model.pricing.RegularPricing;

import java.util.Objects;

public class Movie {
    @Deprecated
    public static final Pricing CHILDRENS = new ChildrenPricing(); // new StandardPricing(1.5, 1.5, 3 )
    @Deprecated
    public static final Pricing REGULAR = new RegularPricing(); // new StandardPricing(2, 1.5, 2 )
    @Deprecated
    public static final Pricing NEW_RELEASE = new NewReleasePricing();

    private final String _title;
    @Deprecated
    private int _priceCode;
    private Pricing _pricing;

    public Movie(String title, Pricing pricing) {
        _title = title;
        _pricing = pricing;
    }

    @Deprecated
    public int getPriceCode() {
        return _priceCode;
    }

    @Deprecated
    public void setPriceCode(int priceCode) {
        _priceCode = priceCode;
    }

    public String getTitle() {
        return _title;
    }

    public Pricing getPricing() {
        return _pricing;
    }

    public void setPricing(Pricing pricing) {
        this._pricing = pricing;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Movie movie = (Movie) o;
        return _title.equals(movie._title) && _pricing.equals(movie._pricing);
    }

    @Override
    public int hashCode() {
        return Objects.hash(_title, _pricing);
    }
}
