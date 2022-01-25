package refactored.model;
import java.util.*;

public class Customers
{
    private String _name;
    private Vector _rentals = new Vector();
    public Customers(String name)
    {
	_name=name;
    }
    public void addRental(Rental rental)
    {
	_rentals.addElement(rental);
    }
    public String getName()
    {
	return _name;
    }
    public String statement()
    {
	double totalAmount = 0;
	int frequentRenterPoints = 0;
	Enumeration rentals=_rentals.elements();
	String result = "Rental Record for "+getName()+"\n";
	while (rentals.hasMoreElements())
	    {
		double thisAmount=0;
		Rental each=(Rental) rentals.nextElement();
		switch (each.getMovie().getPriceCode())
		    {
		    case Movie.REGULAR:
			thisAmount+=2;
			if (each.getDaysRented()>2)
			    {
				thisAmount+=(each.getDaysRented()-2)*1.5;
			    }
			break;
		    case Movie.NEW_RELEASE:
			thisAmount += each.getDaysRented()*3;
			break;
		    case Movie.CHILDRENS:
			thisAmount += 1.5;
			if(each.getDaysRented() > 3)
			    thisAmount+= (each.getDaysRented()-3)*1.5;
			break;
		    }
		frequentRenterPoints++;
		if( (each.getMovie().getPriceCode()== Movie.NEW_RELEASE)
		   && (each.getDaysRented()>1)) 
		    frequentRenterPoints++;
		result +="\t" + each.getMovie().getTitle()+"\t"+
		    String.valueOf(thisAmount) +" \n";
		totalAmount+=thisAmount;
	    }
	result += "Amount owned is " + String.valueOf(totalAmount) +
	    "\n";
	result += "You earned " + String.valueOf(frequentRenterPoints) +
	    " frequent renter points";
	return result;
	 
    }
}
 