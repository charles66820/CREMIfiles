package utile;

import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Parent;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;
import model.Rental;

import java.util.LinkedList;

public class FxStatementBuilder implements StatementBuilder {
    private String _name;
    private LinkedList<Rental> _rentals = new LinkedList<>();

    @Override
    public void setName(String name) {
        this._name = name;
    }

    @Override
    public void setRental(LinkedList<Rental> rentals) {
        this._rentals = rentals;
    }

    public Parent getResult() {
        // Build table
        GridPane table = new GridPane();
        table.setGridLinesVisible(true);
        // Adjust the grid style
        table.setPadding(new Insets(0, 10, 0, 10));
        table.setAlignment(Pos.CENTER);

        double totalAmount = 0;
        int frequentRenterPoints = 0;
        int row = 0;
        for (Rental rental : _rentals) {
            double thisAmount = rental.amount();
            frequentRenterPoints += rental.frequentRenterPoints();
            totalAmount += thisAmount;

            // txt.setStyle("-fx-font: 24 arial;");
            table.add(new Text(rental.getMovie().getTitle()), 0, row);
            table.add(new Text(String.valueOf(thisAmount)), 1, row);
            row++;
        }

        VBox col = new VBox();
        col.setAlignment(Pos.CENTER);
        col.getChildren().add(new Text("Rental Record for " + _name));
        col.getChildren().add(table);
        col.getChildren().add(new Text("Amount owned is " + totalAmount));
        col.getChildren().add(new Text("You earned " + frequentRenterPoints + " frequent renter points"));
        return col;
    }
}