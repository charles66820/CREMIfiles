package oldVersion.model;

public interface PriceCode extends Cloneable {
	double getAmount(int days);
	int getRenterPoints(int days);
	PriceCode clone();
}