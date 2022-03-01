package oldVersion.model;

import java.util.Vector;

public class Customer
{
	private String _name;
	private final Vector<Rental> _rentals = new Vector<>();

	public Customer(String name)
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

	public String statement(String endl, String tab)
	{
		String result = "Rental Record for "+getName()+ endl;
		for(Rental each : _rentals)
		{
			result += 
					tab + each.getMovie().getTitle() 
					+ tab + String.valueOf(each.getAmount()) 
					+ " " + endl;
		}

		result += "Amount owned is " + String.valueOf(getTotalAmount()) +
				endl;
		result += "You earned " + String.valueOf(getRenterPoints()) +
				" frequent renter points";
		return result;
	}

	public String statementHTML()
	{
		StringBuilder result = new StringBuilder();
		result.append("<html><body>\n");
		result.append(statement("<br>", "&nbsp;&nbsp;"));
		result.append("</body></html>");
		return result.toString();
	}

	public String statement()
	{
		return statement("\n","\t");
	}

/*
	public void statement(StatementBuilder bld)
	{
		bld.createNewStatement();
		bld.buildHeader(getName());

		for(Rental each : getRentals())
		{
			bld.buildRental(each);
		}

		bld.buildAmount(getTotalAmount());
		bld.buildRenterPoints(getRenterPoints());

		return bld.getStatement();
	}
*/
	
	double getTotalAmount() 
	{
		double res = 0;
		for(Rental each : _rentals)
		{
			res += each.getAmount();
		}
		return res;
	}

	int getRenterPoints()
	{
		int res = 0;
		for(Rental each : _rentals)
		{
			res += each.getRenterPoints();
		}
		return res;
	}
}
