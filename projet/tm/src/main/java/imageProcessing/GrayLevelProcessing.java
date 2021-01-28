package imageProcessing;

import io.scif.img.ImgIOException;
import io.scif.img.ImgOpener;
import io.scif.img.ImgSaver;
import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import utils.DebugInfo;
import java.io.File;

public class GrayLevelProcessing {

    //ali.larbi@u-bordeaux.fr
    public static void threshold(Img<UnsignedByteType> img, int t) {
        final RandomAccess<UnsignedByteType> r = img.randomAccess();

        final int iw = (int) img.max(0);
        final int ih = (int) img.max(1);

        for (int x = 0; x <= iw; ++x) {
            for (int y = 0; y <= ih; ++y) {
                r.setPosition(x, 0);
                r.setPosition(y, 1);
                final UnsignedByteType val = r.get();
                if (val.get() < t)
                    val.set(0);
                else
                    val.set(255);
            }
        }

    }

    public static void fillBrightnessImageRandomAccess(Img<UnsignedByteType> img, int delta) {
        final RandomAccess<UnsignedByteType> r = img.randomAccess();

        final int iw = (int) img.max(0);
        final int ih = (int) img.max(1);

        for (int x = 0; x <= iw; ++x) {
            for (int y = 0; y <= ih; ++y) {
                // Place cursor
                r.setPosition(x, 0);
                r.setPosition(y, 1);
                final UnsignedByteType val = r.get();
                val.set(Math.min(val.get() + delta, 255));
            }
        }
    }

    public static void fillBrightnessImageCursor(Img<UnsignedByteType> img, int delta) {
        final Cursor<UnsignedByteType> cursor = img.cursor();

        while (cursor.hasNext()) {
            cursor.fwd();
            final UnsignedByteType val = cursor.get();
            val.set(Math.min(val.get() + delta, 255));
        }
    }

    public static void contrastImage(Img<UnsignedByteType> img) {
        final Cursor<UnsignedByteType> cursor = img.cursor();

        int min = 255;
        int max = 0;

        while (cursor.hasNext()) {
            cursor.fwd();
            min = Math.min(cursor.get().get(), min);
            max = Math.max(cursor.get().get(), max);
        }

        cursor.reset();

        while (cursor.hasNext()) {
            cursor.fwd();
            final UnsignedByteType val = cursor.get();
            val.set(Math.max(Math.min((255 * (val.get() - min)) / (max - min), 255), 0));
        }
    }

    public static void contrastImage(Img<UnsignedByteType> img, int resMin, int resMax) {
        final Cursor<UnsignedByteType> cursor = img.cursor();

        int min = 255;
        int max = 0;

        while (cursor.hasNext()) {
            cursor.fwd();
            min = Math.min(cursor.get().get(), min);
            max = Math.max(cursor.get().get(), max);
        }

        cursor.reset();

        while (cursor.hasNext()) {
            cursor.fwd();
            final UnsignedByteType val = cursor.get();
            val.set(Math.max(Math.min((255 * (val.get() - min)) / (max - min), resMax), resMin));
        }
    }

    public static void contrastImageWithLut(Img<UnsignedByteType> img, int resMin, int resMax) {
        final Cursor<UnsignedByteType> cursor = img.cursor();

        int min = 255;
        int max = 0;

        while (cursor.hasNext()) {
            cursor.fwd();
            min = Math.min(cursor.get().get(), min);
            max = Math.max(cursor.get().get(), max);
        }

        cursor.reset();

        int[] lut = new int[256];
        for (int i = 0; i < 256; i++)
            lut[i] = Math.max(Math.min((255 * (i - min)) / (max - min), resMax), resMin);

        while (cursor.hasNext()) {
            cursor.fwd();
            final UnsignedByteType val = cursor.get();
            val.set(lut[val.get()]);
        }
    }

    public static int histogram(Img<UnsignedByteType> img, int k) {
        int r = 0;
        final Cursor<UnsignedByteType> cursor = img.cursor();
        while (cursor.hasNext()) {
            cursor.fwd();
            final UnsignedByteType val = cursor.get();
            if (val.get() == k) r++;
        }
        return r;
    }

    public static int[] histogramComlet(Img<UnsignedByteType> img) {
        int[] r = new int[256];
        final Cursor<UnsignedByteType> cursor = img.cursor();
        while (cursor.hasNext()) {
            cursor.fwd();
            final UnsignedByteType val = cursor.get();
            r[val.get()]++;
        }
        return r;
    }

    public static int cumulatedHistogram(Img<UnsignedByteType> img, int k) {
        int r = 0;
        for (int i = 0; i < k; i++)
            r += histogram(img, i);
        return r;
    }

    public static int cumulatedHistogramWithLut(Img<UnsignedByteType> img, int k) {
        int r = 0;
        int[] lut = histogramComlet(img);
        for (int i = 0; i < k; i++)
            r += lut[i];
        return r;
    }

    public static void contrastImageWithHistogram(Img<UnsignedByteType> img, int N) {
        int[] c = histogramComlet(img);
        final Cursor<UnsignedByteType> cursor = img.cursor();
        while (cursor.hasNext()) {
            cursor.fwd();
            final UnsignedByteType val = cursor.get();
            val.set((c[val.get()] * 255) / N);
        }
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
        final Img<UnsignedByteType> input = (Img<UnsignedByteType>) imgOpener.openImgs(filename, factory).get(0);
        final Img<UnsignedByteType> defautInput = input.copy();
        imgOpener.context().dispose();

        // process image
        long starTime, endTime;
        //threshold(input, 128);

		/*
		starTime = System.nanoTime();
		fillBrightnessImageRandomAccess(input, 50);
		endTime = System.nanoTime();
		System.out.println("fillBrightnessImageRandomAccess (in " + (endTime - starTime) + "ms)");//*/

        /*
		starTime = System.nanoTime();
		fillBrightnessImageCursor(input, 50);
		endTime = System.nanoTime();
		System.out.println("fillBrightnessImageRandomAccess (in " + (endTime - starTime) + "ms)");//*/

        /*
        starTime = System.nanoTime();
        contrastImage(input);
        endTime = System.nanoTime();
        System.out.println("contrastImage (in " + (endTime - starTime) + "ns)");//*/

        /*
        starTime = System.nanoTime();
        contrastImage(input, 0, 255);
        endTime = System.nanoTime();
        System.out.println("contrastImage with min max (in " + (endTime - starTime) + "ns)");//*/

        /*
        starTime = System.nanoTime();
        contrastImageWithLut(input, 0, 255);
        endTime = System.nanoTime();
        System.out.println("contrastImageWithLut with min max (in " + (endTime - starTime) + "ns)");//*/

        /*
        starTime = System.nanoTime();
        int h = histogram(input, 0);
        endTime = System.nanoTime();
        System.out.println("histogram for 0 is " + h + " (in " + (endTime - starTime) + "ns)");//*/

        /*
        starTime = System.nanoTime();
        int hc = cumulatedHistogram(input, 100);
        endTime = System.nanoTime();
        System.out.println("cumulatedHistogram " + hc + " (in " + (endTime - starTime) + "ns)");//*/

        /*
        starTime = System.nanoTime();
        int hclut = cumulatedHistogramWithLut(input, 100);
        endTime = System.nanoTime();
        System.out.println("cumulatedHistogramWithLut " + hclut + " (in " + (endTime - starTime) + "ns)");//*/

        //*
        starTime = System.nanoTime();
        contrastImageWithHistogram(input, 20);
        endTime = System.nanoTime();
        System.out.println("contrastImageWithHistogram with N = 20 (in " + (endTime - starTime) + "ns)");//*/

        DebugInfo.showDebugInfo(defautInput, input, histogramComlet(defautInput), histogramComlet(input));

        // save output image
        final String outPath = args[1];
        File path = new File(outPath);
        // clear the file if it already exists.
        if (path.exists()) {
            path.delete();
        }
        ImgSaver imgSaver = new ImgSaver();
        imgSaver.saveImg(outPath, input);
        imgSaver.context().dispose();
        System.out.println("Image saved in: " + outPath);
    }

}
