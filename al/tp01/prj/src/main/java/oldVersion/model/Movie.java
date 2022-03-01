package oldVersion.model;

public class Movie {
    public static final PriceCode CHILDRENS   = new StandardPricing(1.5, 3, 1.5);
    public static final PriceCode REGULAR     = new StandardPricing(2  , 2, 1.5);
    public static final PriceCode NEW_RELEASE = new NewReleasePricing();

    private String _title;
    private PriceCode _priceCode;

    public Movie(String title, PriceCode priceCode)
    {
        _title = title;
        _priceCode = priceCode;
    }

    public PriceCode getPriceCode()
    {
        return _priceCode;
    }

    public void setPriceCode(PriceCode priceCode)
    {
        _priceCode = priceCode;
    }

    public String getTitle()
    {
        return _title;
    }
}
