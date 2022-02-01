package refactored.model;

public class RegularPricing extends StandardPricing implements Cloneable {
    public RegularPricing() {
        super(2, 1.5, 2);
    }

    @Override
    public RegularPricing clone() {
        return (RegularPricing) super.clone();
    }
}
