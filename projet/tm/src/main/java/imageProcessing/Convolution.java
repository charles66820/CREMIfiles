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
        DebugInfo.showDebugInfo(input, output, null, null);
    }

    /**
     * Question 2.1
     */
    public static void convolution(final Img<UnsignedByteType> input, final Img<UnsignedByteType> output, int[][] kernel) {

    }

    /**
     * Question 2.3
     */
    public static void gaussFilterImgLib(final Img<UnsignedByteType> input, final Img<UnsignedByteType> output) {

    }

    public static void main(final String[] args) throws ImgIOException, IncompatibleTypeException {

        // load image
        if (args.length < 2) {
            System.out.println("missing input or output image filename");
            System.exit(-1);
        }
        final String filename = args[0];
        final ArrayImgFactory<UnsignedByteType> factory = new ArrayImgFactory<>(new UnsignedByteType());
        final ImgOpener imgOpener = new ImgOpener();
        final Img<UnsignedByteType> input = (Img<UnsignedByteType>) imgOpener.openImgs(filename, factory).get(0);
        imgOpener.context().dispose();

        // output image of same dimensions
        final Dimensions dim = input;
        final Img<UnsignedByteType> output = factory.create(dim);

        // mean filter
        //meanFilterSimple(input, output);
        //meanFilterWithBorders(input, output, 1);
        meanFilterWithNeighborhood(input, output, 4);

        final String outPath = args[1];
        File path = new File(outPath);
        if (path.exists()) {
            path.delete();
        }
        ImgSaver imgSaver = new ImgSaver();
        imgSaver.saveImg(outPath, output);
        imgSaver.context().dispose();
        System.out.println("Image saved in: " + outPath);
    }

}
