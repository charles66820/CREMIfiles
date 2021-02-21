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

    public static void rgbToHsv(int r, int g, int b, float[] hsv){
        //HSV 0: t 1: S 2: v
        float rf = r / 255f;
        float gf = g / 255f;
        float bf = b / 255f;

        float max = rf;

        if(gf > rf && gf > bf)
            max = gf;
        if(bf > rf && bf > gf)
            max = bf;

        float min = gf;

        if(rf < gf && rf < bf)
            min = rf;
        if(bf < rf && bf < gf)
            min = bf;

        if(max == min){
            hsv[0] = 0;
        }else if(max == rf){
            hsv[0] = (60 * (gf - bf)/(max - min) + 360) % 360;
        }else if(max == gf){
            hsv[0] = 60 * (bf -rf)/(max - min) + 120;
        }else if(max == bf){
            hsv[0] = 60 * (rf - gf)/(max - min) + 240;
        }

        hsv[1] = max == 0 ? 0 : 1 - (min/max);

        hsv[2] = max;
    }

    public static void hsvToRgb(float h, float s, float v, int[] rgb){
        //RGB 0:R 1:G 2:B
        float chroma = v * s;
        float hprime = h / 60;
        float x = chroma * (1 - Math.abs(hprime % 2 - 1));

        float r = 0,g = 0,b = 0;

        if(hprime >= 0 && hprime <= 1){
            r = chroma;
            g = x;
        }else if(hprime <= 2){
            r = x;
            g = chroma;
        }else if(hprime <= 3){
            g = chroma;
            b = x;
        }else if(hprime <= 4){
            g = x;
            b = chroma;
        }else if(hprime <= 5){
            r = x;
            b = chroma;
        }else if(hprime <= 6){
            r = chroma;
            b = x;
        }

        rgb[0] = Math.round ((r + (v - chroma)) * 255);
        rgb[1] = Math.round ((g + (v - chroma)) * 255);
        rgb[2] = Math.round ((b + (v - chroma)) * 255);
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
            /*System.out.println(
                    "r : " + r.get() +
                            " g : " + g.get() +
                            " b : " + b.get() +
                            " r : " + rgb[0] +
                            " g : " + rgb[1] +
                            " b : " + rgb[2]
            );*/

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

        DebugInfo.addForDebugInfo("colorToGray", defautInput, input);
        input = defautInput.copy(); // Reset input

        //*
        starTime = System.nanoTime();
        changeHue(input, 270);
        endTime = System.nanoTime();
        System.out.println("changeHue with 270 (in " + ((endTime - starTime) / 1000000) + "ms " + (endTime - starTime) + "ms)");
        saveImage(input, "changeHue", outPath);//*/

        DebugInfo.addForDebugInfo("changeHue", defautInput, input);
        DebugInfo.showDebugInfo();
    }
}
