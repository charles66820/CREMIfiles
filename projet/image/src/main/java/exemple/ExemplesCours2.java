package exemple;

import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import io.scif.img.ImgIOException;
import io.scif.img.ImgOpener;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.view.Views;
import net.imglib2.view.ExtendedRandomAccessibleInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.view.IntervalView;

/*
Exemples de cours - outils pour la convolution (traitement des bords, voisinage)
*/
public class ExemplesCours2 {

    public static void testExtend(Img<UnsignedByteType> input) {
        final ExtendedRandomAccessibleInterval<UnsignedByteType, Img<UnsignedByteType>> extendedView = Views.extendZero(input);
        final RandomAccess<UnsignedByteType> in = extendedView.randomAccess();
        final long[] dimensions = new long[2];
        input.dimensions(dimensions);
        System.out.println("Dimensions de l'image initiale : " + dimensions[0] + " x " + dimensions[1]);
        // extendedView.dimensions(dimensions); -> méthode non définie, on ne peut pas non plus utiliser un Cursor sur cette vue

        int x = 4000;
        int y = 4000;
        in.setPosition(x, 0);
        in.setPosition(y, 1);
        System.out.println("Valeur au point " + x + " " + y + " : " + in.get().get() );

    }

    public static void testExpand(Img<UnsignedByteType> input) {
        final IntervalView<UnsignedByteType> expandedView = Views.expandMirrorDouble(input, 1, 1 );

        final RandomAccess<UnsignedByteType> in = expandedView.randomAccess();
        final long[] dimensions = new long[2];
        input.dimensions(dimensions);
        System.out.println("Dimensions de l'image initiale : " + dimensions[0] + " x " + dimensions[1]);
        int x = 0;
        int y = (int) (dimensions[1]-1);

        expandedView.dimensions(dimensions);
        System.out.println("Dimensions de l'image expansée : " + dimensions[0] + " x " + dimensions[1]);


        in.setPosition(x, 0);
        in.setPosition(y, 1);
        System.out.println("Valeur au point " + x + " " + y + " : " + in.get().get() );

        y = y + 1;
        in.setPosition(x, 0);
        in.setPosition(y, 1);
        System.out.println("Valeur au point " + x + " " + y + " : " + in.get().get() );

    }

    public static void testSubPart(Img<UnsignedByteType> input) {
        RandomAccessibleInterval< UnsignedByteType > upperLeftPart =
                Views.interval( input, new long[] { 0, 0 }, new long[]{ 2, 2} );
        final Cursor< UnsignedByteType > c = Views.iterable( upperLeftPart ).cursor();
        while (c.hasNext()){
            c.fwd();
            System.out.print(c.get() + " ");
        }
        System.out.println();
    }



    public static void main(final String[] args) throws ImgIOException, IncompatibleTypeException {


        // load image
        if (args.length < 1) {
            System.out.println("missing input image filename");
            System.exit(-1);
        }
        final String filename = args[0];
        final ArrayImgFactory<UnsignedByteType> factory = new ArrayImgFactory<>(new UnsignedByteType());
        final Img<UnsignedByteType> input = (Img<UnsignedByteType>) new ImgOpener().openImgs(filename, factory).get(0);



        testExtend(input);
        testExpand(input);
        testSubPart(input);


    }

}