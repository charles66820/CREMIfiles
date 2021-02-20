package imageProcessing;

import io.scif.img.ImgIOException;
import io.scif.img.ImgOpener;
import io.scif.img.ImgSaver;
import net.imglib2.Cursor;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.view.Views;
import utils.DebugInfo;

import java.io.File;

public class Conversion {
    public static void color2gray(Img<UnsignedByteType> img) {

        /*int red = 0;
        int green = 1;
        int blue = 2;
        Views.hyperSlices(img, red, green, blue);*/

        final Cursor<UnsignedByteType> cursor = img.cursor();

        while (cursor.hasNext()) {
            cursor.fwd();
            final UnsignedByteType val = cursor.get();

            
            UnsignedByteType R = cursor.get();
            UnsignedByteType G = cursor.get();
            UnsignedByteType B = cursor.get();
            val.set((int) ((0.3 * R.get()) + (0.59 * G.get()) + (0.11 * B.get())));
        }
    }

    public static void saveImage(Img<UnsignedByteType> img, String name, String outFolderPath) {
        String outPath = outFolderPath + "/" + name + ".tif";
        File path = new File(outPath);
        // clear the file if it already exists.
        if (path.exists()) {
            path.delete();
        }
        ImgSaver imgSaver = new ImgSaver();
        imgSaver.saveImg(outPath, img);
        imgSaver.context().dispose();
        System.out.println("Image saved in: " + outPath);
    }

    public static void main(final String[] args) throws ImgIOException, IncompatibleTypeException {
        // load image
        if (args.length < 2) {
            System.out.println("missing input and/or output image filenames");
            System.exit(-1);
        }

        final String filename = args[0];
        if (!new File(filename).exists()) {
            System.err.println("File '" + filename + "' does not exist");
            System.exit(-1);
        }

        final ArrayImgFactory<UnsignedByteType> factory = new ArrayImgFactory<>(new UnsignedByteType());
        final ImgOpener imgOpener = new ImgOpener();
        Img<UnsignedByteType> input = imgOpener.openImgs(filename, factory).get(0);
        final Img<UnsignedByteType> defautInput = input.copy();
        imgOpener.context().dispose();

        final String outPath = args[1];

        // process image
        long starTime, endTime;
        starTime = System.nanoTime();
        color2gray(input);
        endTime = System.nanoTime();
        System.out.println("color2gray (in " + (endTime - starTime) + "ms)");
        saveImage(input, "color2gray", outPath);//*/

        input = defautInput.copy(); // Reset input
        /*
        starTime = System.nanoTime();
        rgb2hsv(input, 50);
        endTime = System.nanoTime();
        System.out.println("rgb2hsv (in " + (endTime - starTime) + "ms)");
        saveImage(input, "rgb2hsv", outPath);//*/

        DebugInfo.showDebugInfo(defautInput, input, null, null);
    }
}
