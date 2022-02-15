import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;
import javafx.stage.Stage;
/**
 *
 * @author auber
 *  Attention il faut rendre la "library" javafx accessible
 *  http://stackoverflow.com/a/32062263/1495627
 */
public class UIMain extends Application {

    @Override
    public void start(Stage primaryStage) {

        // Build a first grid
        GridPane grid1 = new GridPane();
        grid1.setGridLinesVisible(true); // Useful for debug

        // Create and add text to the grid
        Text header = new Text("Number one");
        grid1.add(header, 0, 0);

        // This text is adjusted through CSS
        Text another = new Text("Number two");
        another.setStyle("-fx-font: 24 arial;");
        grid1.add(another, 1, 1);

        // Build a second grid
        GridPane grid2 = new GridPane();
        grid2.setGridLinesVisible(true);

        // Adjust the grid style
        grid2.setHgap(10);
        grid2.setVgap(10);
        grid2.setPadding(new Insets(0, 10, 0, 10));

        // Simpler API only for rows
        grid2.addRow(0, new Text("Row one"));
        grid2.addRow(1, new Text("Row two"));
        grid2.addColumn(1, new Text("Col three"));

        // Put the two grids in a column
        VBox col = new VBox();
        col.setAlignment(Pos.CENTER);
        col.getChildren().add(grid1);
        col.getChildren().add(grid2);

        // Also add a text which will be centered
        col.getChildren().add(new Text("Hello JavaFX"));

        // JavaFX uses Scenes for content
        Scene scene = new Scene(col, 200, 200);

        // Scenes are displayed by a Stage
        primaryStage.setTitle("Hello World!");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
} 