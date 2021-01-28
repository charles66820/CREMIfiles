package utils;

import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import net.imglib2.RandomAccess;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.integer.UnsignedByteType;

public class DebugInfo extends Application {
    private static Image imgSrc;
    private static Image imgDest;
    private static int[] histogramSrc;
    private static int[] histogramDest;

    @Override
    public void start(Stage stage) {
        Group root = new Group();
        if (imgSrc != null) {
            ImageView imageView = new ImageView(imgSrc);
            imageView.setX(25);
            imageView.setY(25);
            imageView.setFitHeight(150);
            imageView.setFitWidth(250);
            imageView.setPreserveRatio(true);
            root.getChildren().add(imageView);
        }
        if (imgSrc != null) {
            ImageView imageView1 = new ImageView(imgDest);
            imageView1.setX(300);
            imageView1.setY(25);
            imageView1.setFitHeight(150);
            imageView1.setFitWidth(250);
            imageView1.setPreserveRatio(true);
            root.getChildren().add(imageView1);
        }

        if (histogramSrc != null) {
            VBox vbox1 = new VBox();
            vbox1.setLayoutX(0);
            vbox1.setLayoutY(175);
            vbox1.setMaxHeight(150);
            vbox1.setMaxWidth(260);

            final CategoryAxis xAxis1 = new CategoryAxis();
            xAxis1.setLabel("k");
            xAxis1.setStyle("-fx-tick-label-font-size:0.6em;");
            final NumberAxis yAxis2 = new NumberAxis();
            yAxis2.setLabel("h(k)");
            yAxis2.setStyle("-fx-tick-label-font-size:0.6em;");

            final BarChart<String, Number> bc1 = new BarChart<>(xAxis1, yAxis2);
            XYChart.Series<String, Number> histogramSrcChart = new XYChart.Series<>();
            for (int i = 0; i < 256; i++)
                histogramSrcChart.getData().add(new XYChart.Data<>("" + i, histogramSrc[i]));
            bc1.getData().add(histogramSrcChart);
            vbox1.getChildren().add(bc1);
            root.getChildren().add(vbox1);
        }

        if (histogramDest != null) {
            VBox vbox2 = new VBox();
            vbox2.setLayoutX(280);
            vbox2.setLayoutY(175);
            vbox2.setMaxHeight(150);
            vbox2.setMaxWidth(260);

            final CategoryAxis xAxis2 = new CategoryAxis();
            xAxis2.setLabel("k");
            xAxis2.setStyle("-fx-tick-label-font-size:0.6em;");
            final NumberAxis yAxis2 = new NumberAxis();
            yAxis2.setLabel("h(k)");
            yAxis2.setStyle("-fx-tick-label-font-size:0.6em;");

            final BarChart<String, Number> bc2 = new BarChart<>(xAxis2, yAxis2);
            XYChart.Series<String, Number> histogramDestChart = new XYChart.Series<>();
            for (int i = 0; i < 256; i++)
                histogramDestChart.getData().add(new XYChart.Data<>("" + i, histogramDest[i]));
            bc2.getData().add(histogramDestChart);
            vbox2.getChildren().add(bc2);
            root.getChildren().add(vbox2);
        }

        Scene scene = new Scene(root, 575, 400);
        stage.setScene(scene);
        stage.setResizable(false);
        stage.show();
    }

    public static void showDebugInfo(Img<UnsignedByteType> imgSrc, Img<UnsignedByteType> imgDest, int[] histogramSrc, int[] histogramDest) {
        if (imgSrc != null) DebugInfo.imgSrc = ImgUnsignedByteType2Image(imgSrc);
        if (imgDest != null) DebugInfo.imgDest = ImgUnsignedByteType2Image(imgDest);
        DebugInfo.histogramSrc = histogramSrc;
        DebugInfo.histogramDest = histogramDest;
        Application.launch(DebugInfo.class);
    }

    public static void showDebugInfo(Img<UnsignedByteType> imgSrc, Img<UnsignedByteType> imgDest) {
        showDebugInfo(imgSrc, imgDest, null, null);
    }

    public static void showDebugInfo(Img<UnsignedByteType> imgSrc, int[] histogramSrc) {
        showDebugInfo(imgSrc, null, histogramSrc, null);
    }

    public static void showDebugInfo(Img<UnsignedByteType> imgSrc) {
        showDebugInfo(imgSrc, null, null, null);
    }

    private static Image ImgUnsignedByteType2Image(Img<UnsignedByteType> img) {
        final RandomAccess<UnsignedByteType> r = img.randomAccess();

        final int iw = (int) img.max(0);
        final int ih = (int) img.max(1);

        WritableImage wr = new WritableImage(iw, ih);
        PixelWriter pw = wr.getPixelWriter();
        for (int x = 0; x < iw; x++)
            for (int y = 0; y < ih; y++) {
                r.setPosition(x, 0);
                r.setPosition(y, 1);
                pw.setColor(x, y, Color.grayRgb(r.get().get()));
            }
        return new ImageView(wr).getImage();
    }

}