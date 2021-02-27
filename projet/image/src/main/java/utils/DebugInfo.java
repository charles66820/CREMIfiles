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
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

import java.util.LinkedList;
import java.util.List;

class ImgResult {
    public final String title;
    public final Image imgSrc;
    public final Image imgDest;
    public final int[] histogramSrc;
    public final int[] histogramDest;

    public ImgResult(String title, Image imgSrc, Image imgDest, int[] histogramSrc, int[] histogramDest) {
        this.title = title;
        this.imgSrc = imgSrc;
        this.imgDest = imgDest;
        this.histogramSrc = histogramSrc;
        this.histogramDest = histogramDest;
    }

}

public class DebugInfo extends Application {
    private static final List<ImgResult> imgResults = new LinkedList<>();

    public static void addImgRes(String title, Image imgSrc, Image imgDest, int[] histogramSrc, int[] histogramDest) {
        imgResults.add(new ImgResult(title, imgSrc, imgDest, histogramSrc, histogramDest));
    }

    @Override
    public void start(Stage stage) {
        for (ImgResult imgRes : imgResults) {
            Stage newStage = new Stage();
            newStage.setTitle(imgRes.title);
            this.newWindows(newStage, imgRes);
        }
    }

    public void newWindows(Stage stage, ImgResult imgRes) {

        Group root = new Group();
        if (imgRes.imgSrc != null) {
            ImageView imageView = new ImageView(imgRes.imgSrc);
            if (imgRes.imgSrc.getHeight() > imgRes.imgSrc.getWidth()) {
                imageView.setX(25);
                imageView.setY(25);
                imageView.setFitHeight(350);
                imageView.setFitWidth(250);
            } else {
                imageView.setX(25);
                imageView.setY(25);
                imageView.setFitHeight(350);
                imageView.setFitWidth(500);
            }
            imageView.setPreserveRatio(true);
            root.getChildren().add(imageView);
        }
        if (imgRes.imgDest != null) {
            ImageView imageView1 = new ImageView(imgRes.imgDest);
            if (imgRes.imgSrc.getHeight() > imgRes.imgSrc.getWidth()) {
                imageView1.setX(300);
                imageView1.setY(25);
                imageView1.setFitHeight(350);
                imageView1.setFitWidth(250);
            } else {
                imageView1.setX(550);
                imageView1.setY(25);
                imageView1.setFitHeight(350);
                imageView1.setFitWidth(500);
            }
            imageView1.setPreserveRatio(true);
            root.getChildren().add(imageView1);
        }

        if (imgRes.histogramSrc != null) {
            VBox vbox1 = new VBox();
            vbox1.setLayoutX(0);
            vbox1.setLayoutY(375);
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
                histogramSrcChart.getData().add(new XYChart.Data<>("" + i, imgRes.histogramSrc[i]));
            bc1.getData().add(histogramSrcChart);
            vbox1.getChildren().add(bc1);
            root.getChildren().add(vbox1);
        }

        if (imgRes.histogramDest != null) {
            VBox vbox2 = new VBox();
            vbox2.setLayoutX(280);
            vbox2.setLayoutY(375);
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
                histogramDestChart.getData().add(new XYChart.Data<>("" + i, imgRes.histogramDest[i]));
            bc2.getData().add(histogramDestChart);
            vbox2.getChildren().add(bc2);
            root.getChildren().add(vbox2);
        }

        Scene scene = new Scene(root, 1075, 600);
        stage.setScene(scene);
        stage.setResizable(false);
        stage.show();
    }

    public static void addForDebugInfo(String title, Img<UnsignedByteType> imgSrc, Img<UnsignedByteType> imgDest, int[] histogramSrc, int[] histogramDest) {
        addImgRes(title, ImgUnsignedByteType2Image(imgSrc), ImgUnsignedByteType2Image(imgDest), histogramSrc, histogramDest);
    }

    public static void addForDebugInfo(String title, Img<UnsignedByteType> imgSrc, Img<UnsignedByteType> imgDest) {
        addForDebugInfo(title, imgSrc, imgDest, null, null);
    }

    public static void addForDebugInfo(String title, Img<UnsignedByteType> imgSrc, int[] histogramSrc) {
        addForDebugInfo(title, imgSrc, null, histogramSrc, null);
    }

    public static void addForDebugInfo(String title, Img<UnsignedByteType> imgSrc) {
        addForDebugInfo(title, imgSrc, null, null, null);
    }

    private static Image ImgUnsignedByteType2Image(Img<UnsignedByteType> img) {
        if (img == null) return null;

        final int iw = (int) img.max(0);
        final int ih = (int) img.max(1);

        WritableImage wr = new WritableImage(iw, ih);
        PixelWriter pw = wr.getPixelWriter();

        final RandomAccess<UnsignedByteType> ra = img.randomAccess();
        if (img.numDimensions() == 2) {
            for (int x = 0; x < iw; x++)
                for (int y = 0; y < ih; y++) {
                    ra.setPosition(x, 0);
                    ra.setPosition(y, 1);
                    pw.setColor(x, y, Color.grayRgb(ra.get().get()));
                }
        } else {
            for (int x = 0; x < iw; x++)
                for (int y = 0; y < ih; y++) {
                    int[] rgb = new int[(int) img.dimension(2)];
                    for (int z = 0; z < img.dimension(2); z++) {
                        ra.setPosition(x, 0);
                        ra.setPosition(y, 1);
                        ra.setPosition(z, 2);
                        rgb[z] = ra.get().get();
                    }
                    pw.setColor(x, y, Color.rgb(rgb[0], rgb[1], rgb[2]));
                }
        }
        return new ImageView(wr).getImage();
    }

    public static void showDebugInfo() {
        Application.launch(DebugInfo.class);
    }

}