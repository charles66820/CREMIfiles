package oldVersion.model;

public class NewReleasePricing extends StandardPricing {

	public NewReleasePricing() {
		super(0, 0, 3);
	}
	@Override
	public int getRenterPoints(int days) {
		if (days > 1)
			return 2;
		else 
			return 1;
	}

}
