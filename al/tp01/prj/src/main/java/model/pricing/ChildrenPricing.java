package model.pricing;

public class ChildrenPricing extends StandardPricing implements Cloneable {
    public ChildrenPricing() {
        super(1.5, 1.5, 3);
    }

    @Override
    public ChildrenPricing clone() {
        return (ChildrenPricing) super.clone();
    }
}
