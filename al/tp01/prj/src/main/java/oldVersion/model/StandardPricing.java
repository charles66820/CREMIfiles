package oldVersion.model;

public class StandardPricing implements PriceCode {
	private double fixedPrice, daysOfPackage, extraPrice;
	
	public StandardPricing(double fixedPrice, double daysOfPackage, double extraPrice) {
		this.fixedPrice = fixedPrice;
		this.daysOfPackage = daysOfPackage;
		this.extraPrice = extraPrice;
	}

	@Override
	public PriceCode clone() {
		PriceCode tmp = null;
		try {
			tmp = (PriceCode) super.clone();
		}
		catch(CloneNotSupportedException e) {
			System.out.println("Impossible");
			System.exit(-1);
		}
		return tmp;
	}
	
	@Override
	public double getAmount(int days) {
		double val = fixedPrice;
		if (days>daysOfPackage)
		{
			val += (days - daysOfPackage) * extraPrice;
		}
		return val;		
	}

	@Override
	public int getRenterPoints(int days) {
		return 1;
	}

}