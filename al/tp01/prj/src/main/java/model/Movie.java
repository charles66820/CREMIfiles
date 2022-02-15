package model;

import model.pricing.ChildrenPricing;
import model.pricing.NewReleasePricing;
import model.pricing.Pricing;
import model.pricing.RegularPricing;

public class Movie {
    @Deprecated
    public static final Pricing CHILDRENS = new ChildrenPricing();
    @Deprecated
    public static final Pricing REGULAR = new RegularPricing();
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
}
