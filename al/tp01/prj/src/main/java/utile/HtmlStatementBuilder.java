package utile;

import model.Rental;

import java.util.LinkedList;

public class HtmlStatementBuilder implements StatementBuilder {
    private String _name;
    private LinkedList<Rental> _rentals = new LinkedList<>();

    private String statement(String endl, String tab) {
        double totalAmount = 0;
        int frequentRenterPoints = 0;

        StringBuilder result = new StringBuilder("Rental Record for " + _name + endl);
        for (Rental rental : new LinkedList<>(_rentals)) {
            double thisAmount = rental.amount();
            frequentRenterPoints += rental.frequentRenterPoints();

            result.append(tab).append(rental.getMovie().getTitle());
            result.append(tab).append(thisAmount).append(" ").append(endl);
            totalAmount += thisAmount;
        }

        result.append("Amount owned is ").append(totalAmount).append(endl);
        result.append("You earned ").append(frequentRenterPoints).append(" frequent renter points");
        return result.toString();
    }

    @Override
    public void setName(String name) {
        this._name = name;
    }

    @Override
    public void setRental(LinkedList<Rental> rentals) {
        this._rentals = rentals;
    }

    @Override
    public String getResult() {
        return "<html><body>\n" +
                statement("<br>", "&nbsp;&nbsp;") +
                "</body></html>";
    }
}
