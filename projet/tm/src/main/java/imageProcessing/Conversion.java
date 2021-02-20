package imageProcessing;

import io.scif.img.ImgIOException;
import io.scif.img.ImgOpener;
import io.scif.img.ImgSaver;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import utils.DebugInfo;

import java.io.File;

public class Conversion {
    public static void colorToGray(Img<UnsignedByteType> img) {
        final IntervalView<UnsignedByteType> cR = Views.hyperSlice(img, 2, 0); // Dimension 2 channel 0 (red)
        final IntervalView<UnsignedByteType> cG = Views.hyperSlice(img, 2, 1); // Dimension 2 channel 1 (green)
        final IntervalView<UnsignedByteType> cB = Views.hyperSlice(img, 2, 2); // Dimension 2 channel 2 (blue)

        LoopBuilder.setImages(cR, cG, cB).forEachPixel((r, g, b) -> {
            int grayColor = (int) ((0.3 * r.get()) + (0.59 * g.get()) + (0.11 * b.get()));
            r.set(grayColor);
            g.set(grayColor);
            b.set(grayColor);
        });
    }

    static void rgbToHsv(int r, int g, int b, float[] hsv) { // hue (la teinte), saturation, value
        // Hue
        float h;
        float max = Math.max(r, Math.max(g, b));
        float min = Math.min(r, Math.min(g, b));
        float chroma = max - min;
        if (chroma == 0) h = 0;
        else if (max == r) h = (60 * (((g - b) / chroma) + 360)) % 360;
        else if (max == g) h = (60 * (((b - r) / chroma) + 120));
        else h = (60 * (((r - g) / chroma) + 240)); // max == b
        hsv[0] = h;

        // Saturation
        float s;
        if (max == 0) s = 0;
        else s = (1f - (min / max));
        hsv[1] = s;

        // Value
        hsv[2] = max;
    }

    static void hsvToRgb(float h, float s, float v, int[] rgb) {
        float ti = (float) (Math.floor(h / 60f) % 6f);
        float f = (h / 60f) - ti;

        float l = v * (1f - s);
        float m = v * (1f - (f * s));
        float n = v * (1f - ((1f - f) * s));

        float r = 0, g = 0, b = 0;
        switch ((int) ti) {
            case 0:
                r = v;
                g = n;
                b = l;
                break;
            case 1:
                r = m;
                g = v;
                b = l;
                break;
            case 2:
                r = l;
                g = v;
                b = n;
                break;
            case 3:
                r = l;
                g = m;
                b = v;
                break;
            case 4:
                r = n;
                g = l;
                b = v;
                break;
            case 5:
                r = v;
                g = l;
                b = m;
        }

        rgb[0] = (int) r;
        rgb[1] = (int) g;
        rgb[2] = (int) b;
    }

    public static void changeHue(Img<UnsignedByteType> img, int hue) {
        if (hue > 360) return;
        final IntervalView<UnsignedByteType> cR = Views.hyperSlice(img, 2, 0); // Dimension 2 channel 0 (red)
        final IntervalView<UnsignedByteType> cG = Views.hyperSlice(img, 2, 1); // Dimension 2 channel 1 (green)
        final IntervalView<UnsignedByteType> cB = Views.hyperSlice(img, 2, 2); // Dimension 2 channel 2 (blue)

        LoopBuilder.setImages(cR, cG, cB).forEachPixel((r, g, b) -> {
            float[] hsv = new float[3];
            rgbToHsv(r.get(), g.get(), b.get(), hsv);
            hsv[0] = hue;
            int[] rgb = new int[3];
            hsvToRgb(hsv[0], hsv[1], hsv[2], rgb);

            r.set(rgb[0]);
            g.set(rgb[1]);
            b.set(rgb[2]);
        });
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
        //*
        starTime = System.nanoTime();
        colorToGray(input);
        endTime = System.nanoTime();
        System.out.println("colorToGray (in " + ((endTime - starTime) / 1000000) + "ms " + (endTime - starTime) + "ms)");
        saveImage(input, "colorToGray", outPath);//*/

        input = defautInput.copy(); // Reset input

        //*
        starTime = System.nanoTime();
        changeHue(input, 270);
        endTime = System.nanoTime();
        System.out.println("changeHue with 270 (in " + ((endTime - starTime) / 1000000) + "ms " + (endTime - starTime) + "ms)");
        saveImage(input, "changeHue", outPath);//*/

        DebugInfo.showDebugInfo(defautInput, input, null, null);
    }
}
