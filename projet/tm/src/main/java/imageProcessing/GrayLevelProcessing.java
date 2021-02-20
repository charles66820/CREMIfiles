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

    // Fonctionne seulement sur le rouge avec une image couleur car il y a trois canneaux donc il faut partourire les trois image R, G et B
    public static void threshold(Img<UnsignedByteType> img, int t) {
        // use img.numDimensions() and img.dimension() for fix
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

    // Fonctionne seulement sur le rouge avec une image couleur car il y a trois canneaux donc il faut partourire les trois image R, G et B
    public static void fillBrightnessImageRandomAccess(Img<UnsignedByteType> img, int delta) {
        // use img.numDimensions() and img.dimension() for fix
        final RandomAccess<UnsignedByteType> r = img.randomAccess();

        final int iw = (int) img.max(0);
        final int ih = (int) img.max(1);

        if (img.numDimensions() == 2) // For gray
            for (int x = 0; x <= iw; ++x) {
                for (int y = 0; y <= ih; ++y) {
                    // Place cursor
                    r.setPosition(x, 0);
                    r.setPosition(y, 1);
                    final UnsignedByteType val = r.get();
                    val.set(Math.min(val.get() + delta, 255));
                }
            }
        else // Colors
            for (int d = 0; d < img.dimension(2); d++) // For support dimention
                for (int x = 0; x <= iw; ++x) {
                    for (int y = 0; y <= ih; ++y) {
                        // Place cursor
                        r.setPosition(x, 0);
                        r.setPosition(y, 1);
                        r.setPosition(d, 2);
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

    public static int[] histogramComplet(Img<UnsignedByteType> img) {
        int[] r = new int[256];
        final Cursor<UnsignedByteType> cursor = img.cursor();
        while (cursor.hasNext()) {
            cursor.fwd();
            final UnsignedByteType val = cursor.get();
            r[val.get()]++;
        }
        return r;
    }

    // For colors
    public static int[] histogramComplet(Img<UnsignedByteType> img, int dimension) {
        int[] r = new int[256];

        final RandomAccess<UnsignedByteType> ri = img.randomAccess();

        final int iw = (int) img.max(0);
        final int ih = (int) img.max(1);
        for (int x = 0; x <= iw; ++x) {
            for (int y = 0; y <= ih; ++y) {
                // Place cursor
                ri.setPosition(x, 0);
                ri.setPosition(y, 1);
                ri.setPosition(dimension, 2);
                final UnsignedByteType val = ri.get();
                r[val.get()]++;
            }
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
        int[] histogramLut = histogramComplet(img);
        for (int i = 0; i < k; i++)
            r += histogramLut[i];
        return r;
    }

    public static int[] cumulatedHistogramWithLut(Img<UnsignedByteType> img) {
        int[] rLut = new int[256];
        int[] histogramLut = histogramComplet(img);
        for (int i = 0; i < 256; i++)
            for (int j = 0; j < i; j++)
                rLut[i] += histogramLut[j];
        return rLut;
    }

    // For colors
    public static int[] cumulatedColorsHistogramWithLut(Img<UnsignedByteType> img, int dimension) {
        int[] rLut = new int[256];
        int[] histogramLut = histogramComplet(img, dimension);
        for (int i = 0; i < 256; i++)
            for (int j = 0; j < i; j++)
                rLut[i] += histogramLut[j];
        return rLut;
    }

    // Il y a une erreur car on faire l'histogram cumuler sur les 3 canaux au lieu de le faire sur chaque cannaux separement
    public static void contrastImageWithHistogram(Img<UnsignedByteType> img) {
        int N = (int) img.max(0) * (int) img.max(1);

        if (img.numDimensions() == 2) {
            // Gray
            int[] c = cumulatedHistogramWithLut(img);
            final Cursor<UnsignedByteType> cursor = img.cursor();
            while (cursor.hasNext()) {
                cursor.fwd();
                final UnsignedByteType val = cursor.get();
                val.set((c[val.get()] * 255) / N);
            }
        } else { // Colors*/
            final RandomAccess<UnsignedByteType> ri = img.randomAccess();
            final int iw = (int) img.max(0);
            final int ih = (int) img.max(1);

            for (int d = 0; d < img.dimension(2); d++) {
                int[] c = cumulatedColorsHistogramWithLut(img, d);
                for (int x = 0; x <= iw; ++x) {
                    for (int y = 0; y <= ih; ++y) {
                        // Place cursor
                        ri.setPosition(x, 0);
                        ri.setPosition(y, 1);
                        ri.setPosition(d, 2);
                        final UnsignedByteType val = ri.get();
                        val.set((c[val.get()] * 255) / N);
                    }
                }
            }
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
        threshold(input, 128);
        endTime = System.nanoTime();
        System.out.println("threshold (in " + (endTime - starTime) + "ms)");
        saveImage(input, "threshold", outPath);//*/

        input = defautInput.copy(); // Reset input

        //*
        starTime = System.nanoTime();
        fillBrightnessImageRandomAccess(input, 50);
        endTime = System.nanoTime();
        System.out.println("fillBrightnessImageRandomAccess (in " + (endTime - starTime) + "ms)");
        saveImage(input, "fillBrightnessImageRandomAccess", outPath);//*/

        input = defautInput.copy(); // Reset input

        //*
        starTime = System.nanoTime();
        fillBrightnessImageCursor(input, 50);
        endTime = System.nanoTime();
        System.out.println("fillBrightnessImageCursor (in " + (endTime - starTime) + "ms)");
        saveImage(input, "fillBrightnessImageCursor", outPath);//*/

        input = defautInput.copy(); // Reset input

        //*
        starTime = System.nanoTime();
        contrastImage(input);
        endTime = System.nanoTime();
        System.out.println("contrastImage (in " + (endTime - starTime) + "ns)");
        saveImage(input, "contrastImage", outPath);//*/

        input = defautInput.copy(); // Reset input

        //*
        starTime = System.nanoTime();
        contrastImage(input, 0, 255);
        endTime = System.nanoTime();
        System.out.println("contrastImage with min max (in " + (endTime - starTime) + "ns)");
        saveImage(input, "contrastImageMinMax", outPath);//*/

        input = defautInput.copy(); // Reset input

        //*
        starTime = System.nanoTime();
        contrastImageWithLut(input, 0, 255);
        endTime = System.nanoTime();
        System.out.println("contrastImageWithLut with min max (in " + (endTime - starTime) + "ns)");
        saveImage(input, "contrastImageWithLutMinMax", outPath);//*/

        input = defautInput.copy(); // Reset input

        //*
        starTime = System.nanoTime();
        int h = histogram(input, 0);
        endTime = System.nanoTime();
        System.out.println("histogram for 0 is " + h + " (in " + (endTime - starTime) + "ns)");
//*/
        //*
        starTime = System.nanoTime();
        int hc = cumulatedHistogram(input, 100);
        endTime = System.nanoTime();
        System.out.println("cumulatedHistogram " + hc + " (in " + (endTime - starTime) + "ns)");
//*/
        //*
        starTime = System.nanoTime();
        int hclut = cumulatedHistogramWithLut(input, 100);
        endTime = System.nanoTime();
        System.out.println("cumulatedHistogramWithLut " + hclut + " (in " + (endTime - starTime) + "ns)");
//*/
        //*
        starTime = System.nanoTime();
        contrastImageWithHistogram(input);
        endTime = System.nanoTime();
        System.out.println("contrastImageWithHistogram with N = 20 (in " + (endTime - starTime) + "ns)");
        saveImage(input, "contrastImageWithHistogram", outPath);//*/

        DebugInfo.showDebugInfo(defautInput, input, histogramComplet(defautInput), histogramComplet(input));
    }
}
