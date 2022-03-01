import javafx.application.Application;
import javafx.scene.Scene;
import javafx.stage.Stage;
import model.Customer;
import model.Movie;
import model.Rental;
import model.pricing.ChildrenPricing;
import model.pricing.NewReleasePricing;
import model.pricing.RegularPricing;
import utile.FxStatementBuilder;

public class UIMain extends Application {

    @Override
    public void start(Stage primaryStage) {
        Customer c = new Customer("Toto");
        c.addRental(new Rental(new Movie("Rogue One", new NewReleasePricing()), 5));
        c.addRental(new Rental(new Movie("Reine des neiges", new ChildrenPricing()), 7));
        c.addRental(new Rental(new Movie("Star Wars III", new RegularPricing()), 4));

        FxStatementBuilder statementBuilder = new FxStatementBuilder();
        c.completeBuilder(statementBuilder);

        Scene scene = new Scene(statementBuilder.getResult(), 220, 100);
        primaryStage.setTitle(c.getName());
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
} 