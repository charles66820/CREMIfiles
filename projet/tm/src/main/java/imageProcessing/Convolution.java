package imageProcessing;

import net.imglib2.*;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import io.scif.img.ImgIOException;
import io.scif.img.ImgOpener;
import io.scif.img.ImgSaver;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.exception.IncompatibleTypeException;

import java.io.File;

import net.imglib2.view.Views;
import net.imglib2.view.ExtendedRandomAccessibleInterval;
import net.imglib2.view.IntervalView;
import net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.algorithm.neighborhood.RectangleShape;
import net.imglib2.algorithm.neighborhood.Neighborhood;
import net.imglib2.util.Intervals;
import utils.DebugInfo;

public class Convolution {

    /**
     * Question 1.1
     */
    public static void meanFilterSimple(final Img<UnsignedByteType> input, final Img<UnsignedByteType> output) {
        final RandomAccess<UnsignedByteType> in = input.randomAccess();
        final RandomAccess<UnsignedByteType> out = output.randomAccess();

        final int iw = (int) input.max(0);
        final int ih = (int) input.max(1);

        for (int x = 1; x < iw - 1; x++)
            for (int y = 1; y < ih - 1; y++) {
                out.setPosition(x, 0);
                out.setPosition(y, 1);
                int sum = 0;
                for (int fx = x - 1; fx <= x + 1; fx++)
                    for (int fy = y - 1; fy <= y + 1; fy++) {
                        in.setPosition(fx, 0);
                        in.setPosition(fy, 1);
                        sum += in.get().get();
                    }
                out.get().set(sum / 9);
            }
    }

    /**
     * Question 1.2
     */
    public static void meanFilterWithBorders(final Img<UnsignedByteType> input, final Img<UnsignedByteType> output, int size) {
        if (input.numDimensions() != 2) return; // Fix for colorised image

        final int iw = (int) input.max(0);
        final int ih = (int) input.max(1);

        //IntervalView<UnsignedByteType> expandView;
        // Expand zero
        //expandView = Views.expandValue(input, 0, iw, ih);

        // Expand mirror
        //expandView = Views.expandMirrorDouble(input, -1);

        ExtendedRandomAccessibleInterval<UnsignedByteType, Img<UnsignedByteType>> extendedView;
        // Extend with zero
        //extendedView = Views.extendZero(input);

        // Extend mirror
        extendedView = Views.extendMirrorDouble(input);

        final RandomAccess<UnsignedByteType> out = output.randomAccess();

        for (int x = 0; x < iw; x++)
            for (int y = 0; y < ih; y++) {
                // Neighbor filter
                final IterableInterval<UnsignedByteType> neighborView =
                        Views.interval(extendedView, new long[]{x - size, y - size}, new long[]{x + size, y + size});
                // Sum all neighbor values
                final Cursor<UnsignedByteType> cursor = neighborView.cursor();
                int sum = 0;
                while (cursor.hasNext()) {
                    cursor.fwd();
                    sum += cursor.get().get();
                }

                // Set new value to output image
                out.setPosition(x, 0);
                out.setPosition(y, 1);
                out.get().set(sum / (((size * 2) + 1) * ((size * 2) + 1)));
            }
    }

    /**
     * Question 1.3
     */
    public static void meanFilterWithNeighborhood(final Img<UnsignedByteType> input, final Img<UnsignedByteType> output, int size) {
        Interval interval = Intervals.expand(input, -size);

        IntervalView<UnsignedByteType> source = Views.interval(input, interval);
        final Cursor<UnsignedByteType> sourceCursor = source.cursor();

        IntervalView<UnsignedByteType> dest = Views.interval(output, interval);
        final Cursor<UnsignedByteType> destCursor = dest.cursor();

        final RectangleShape shape = new RectangleShape(size, true);

        for (final Neighborhood<UnsignedByteType> localNeighborhood : shape.neighborhoods(source)) {
            final UnsignedByteType sourceValue = sourceCursor.next();
            final UnsignedByteType destValue = destCursor.next();

            int sum = sourceValue.get();
            for (final UnsignedByteType value : localNeighborhood) sum += value.get();
            destValue.set(sum / (((size * 2) + 1) * ((size * 2) + 1)));
        }
    }

    /**
     * Question 2.1
     */
    public static void convolution(final Img<UnsignedByteType> input, final Img<UnsignedByteType> output, int[][] kernel) {
        int size = (kernel.length - 1) / 2;
        Interval interval = Intervals.expand(input, -size);

        IntervalView<UnsignedByteType> source = Views.interval(input, interval);
        final Cursor<UnsignedByteType> sourceCursor = source.cursor();

        IntervalView<UnsignedByteType> dest = Views.interval(output, interval);
        final Cursor<UnsignedByteType> destCursor = dest.cursor();

        final RectangleShape shape = new RectangleShape(size, true);

        int kernelSum = 0;
        for (int[] row : kernel) for (int value : row) kernelSum += value;

        for (final Neighborhood<UnsignedByteType> localNeighborhood : shape.neighborhoods(source)) {
            final UnsignedByteType sourceValue = sourceCursor.next();
            final UnsignedByteType destValue = destCursor.next();

            int sourceX = sourceCursor.getIntPosition(0);
            int sourceY = sourceCursor.getIntPosition(1);
            int sum = sourceValue.get();

            Cursor<UnsignedByteType> localNeighborhoodCursor = localNeighborhood.cursor();
            while(localNeighborhoodCursor.hasNext()) {
                final UnsignedByteType value = localNeighborhoodCursor.next();
                int[] pos = new int[localNeighborhoodCursor.numDimensions()];
                localNeighborhoodCursor.localize(pos);

                int x = pos[0] - (sourceX - size);
                int y = pos[1] - (sourceY - size);

                sum += (value.get() * kernel[x][y]);
            }

            destValue.set(sum / kernelSum);
        }
    }

    /**
     * Question 2.3
     */
    public static void gaussFilterImgLib(final Img<UnsignedByteType> input, final Img<UnsignedByteType> output) {
        Gauss3.gauss((double) 4 / 3, Views.extendMirrorDouble(input), output);
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
            System.out.println("missing input or output image filename");
            System.exit(-1);
        }

        final String filename = args[0];
        if (!new File(filename).exists()) {
            System.err.println("File '" + filename + "' does not exist");
            System.exit(-1);
        }

        final ArrayImgFactory<UnsignedByteType> factory = new ArrayImgFactory<>(new UnsignedByteType());
        final ImgOpener imgOpener = new ImgOpener();
        final Img<UnsignedByteType> input = imgOpener.openImgs(filename, factory).get(0);
        imgOpener.context().dispose();

        final String outPath = args[1];

        // output image of same dimensions
        final Dimensions dim = input;
        final Img<UnsignedByteType> output = factory.create(dim);

        // mean filter
        long starTime, endTime;
        //*
        starTime = System.nanoTime();
        meanFilterSimple(input, output);
        endTime = System.nanoTime();
        System.out.println("meanFilterSimple (in " + ((endTime - starTime) / 1000000) + "ms " + (endTime - starTime) + "ns)");
        saveImage(output, "meanFilterSimple", outPath);//*/
        DebugInfo.addForDebugInfo("meanFilterSimple", input, output, null, null);

        //*
        starTime = System.nanoTime();
        meanFilterWithBorders(input, output, 1);
        endTime = System.nanoTime();
        System.out.println("meanFilterWithBorders (in " + ((endTime - starTime) / 1000000) + "ms " + (endTime - starTime) + "ns)");
        saveImage(output, "meanFilterWithBorders", outPath);//*/
        DebugInfo.addForDebugInfo("meanFilterWithBorders", input, output, null, null);

        //*
        starTime = System.nanoTime();
        meanFilterWithNeighborhood(input, output, 4);
        endTime = System.nanoTime();
        System.out.println("meanFilterWithNeighborhood (in " + ((endTime - starTime) / 1000000) + "ms " + (endTime - starTime) + "ns)");
        saveImage(output, "meanFilterWithNeighborhood", outPath);//*/
        DebugInfo.addForDebugInfo("meanFilterWithNeighborhood", input, output, null, null);

        int[][] kernelOne = new int[][]{
                {1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1}
        };
        int[][] kernel = new int[][]{
                {1, 2, 3, 2, 1},
                {2, 6, 8, 6, 2},
                {3, 8, 10, 8, 3},
                {2, 6, 8, 6, 2},
                {1, 2, 3, 2, 1}
        };

        //*
        starTime = System.nanoTime();
        convolution(input, output, kernelOne);
        endTime = System.nanoTime();
        System.out.println("my gauss convolution with one (in " + ((endTime - starTime) / 1000000) + "ms " + (endTime - starTime) + "ns)");
        saveImage(output, "myGaussConvolutionWithOne", outPath);//*/
        DebugInfo.addForDebugInfo("myGaussConvolutionWithOne", input, output, null, null);

        //*
        starTime = System.nanoTime();
        convolution(input, output, kernel);
        endTime = System.nanoTime();
        System.out.println("my gauss convolution (in " + ((endTime - starTime) / 1000000) + "ms " + (endTime - starTime) + "ns)");
        saveImage(output, "myGaussConvolutionWithKernel", outPath);//*/
        DebugInfo.addForDebugInfo("myGaussConvolutionWithKernel", input, output, null, null);

        //*
        starTime = System.nanoTime();
        gaussFilterImgLib(input, output);
        endTime = System.nanoTime();
        System.out.println("default gauss convolution (in " + ((endTime - starTime) / 1000000) + "ms " + (endTime - starTime) + "ns)");
        saveImage(output, "defaultGaussConvolutionWithKernel", outPath);//*/

        DebugInfo.addForDebugInfo("defaultGaussConvolutionWithKernel", input, output, null, null);
        DebugInfo.showDebugInfo();
    }
}
