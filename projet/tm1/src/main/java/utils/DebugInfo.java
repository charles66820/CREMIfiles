package utils;

import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import net.imglib2.RandomAccess;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.integer.UnsignedByteType;

public class DebugInfo extends Application {
    private static Image imgSrc;
    private static Image imgDest;

    @Override
    public void start(Stage stage) {
        Group root = new Group();
        if (imgSrc != null) {
            ImageView imageView = new ImageView(imgSrc);
            imageView.setX(25);
            imageView.setY(25);
            imageView.setFitHeight(300);
            imageView.setFitWidth(250);
            imageView.setPreserveRatio(true);
            root.getChildren().add(imageView);
        }
        if (imgSrc != null) {
            ImageView imageView1 = new ImageView(imgDest);
            imageView1.setX(300);
            imageView1.setY(25);
            imageView1.setFitHeight(300);
            imageView1.setFitWidth(250);
            imageView1.setPreserveRatio(true);
            root.getChildren().add(imageView1);
        }

        Scene scene = new Scene(root, 575, 400);
        stage.setScene(scene);
        stage.setResizable(false);
        stage.show();
    }

    public static void showDebugInfo(Img<UnsignedByteType> imgSrc, Img<UnsignedByteType> imgDest) {
        if (imgSrc != null) DebugInfo.imgSrc = ImgUnsignedByteType2Image(imgSrc);
        if (imgDest != null)  DebugInfo.imgDest = ImgUnsignedByteType2Image(imgDest);
        Application.launch(DebugInfo.class);
    }

    public static void showDebugInfo(Img<UnsignedByteType> imgSrc) {
        showDebugInfo(imgSrc, null);
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