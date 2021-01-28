package exemple;

import net.imglib2.Cursor;
import net.imglib2.Dimensions;
import net.imglib2.RandomAccess;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import io.scif.img.ImgIOException;
import io.scif.img.ImgOpener;
import io.scif.img.ImgSaver;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.exception.IncompatibleTypeException;
import java.io.File;

/*
Exemples de cours - traitement point par point d'une image en niveau de gris
seuillage basique : un pixel devient noir s'il est inférieur au seuil, blanc sinon
2 versions : avec effet de bord / sans effet de bord
*/
public class OpenProcessSaveGrayLevelImage {

	public static void simpleThreshold (Img<UnsignedByteType> input, int threshold) {
		final Cursor<UnsignedByteType> in = input.cursor();
		
		while (in.hasNext()) {
			in.fwd();
			final int val = in.get().get();
			if (val < threshold)
				in.get().set(0);
			else
				in.get().set(255);
		}
	}

	public static void simpleThreshold(final Img<UnsignedByteType> input, final Img<UnsignedByteType> output, int threshold) {
		final RandomAccess<UnsignedByteType> in = input.randomAccess();
		final Cursor<UnsignedByteType> out = output.localizingCursor();

		while (out.hasNext()) {
			out.fwd();
			in.setPosition(out);
			final int val = in.get().get();
			if (val < threshold)
				out.get().set(0);
			else
				out.get().set(255);
		}
	}

	
	public static void main(final String[] args) throws ImgIOException, IncompatibleTypeException {
		// load image
		if (args.length < 2) {
			System.out.println("missing input or output image filename");
			System.exit(-1);
		} 
		final String filename = args[0];
		final ArrayImgFactory<UnsignedByteType> factory = new ArrayImgFactory<>(new UnsignedByteType());
		final Img<UnsignedByteType> input = (Img<UnsignedByteType>) new ImgOpener().openImgs(filename, factory).get(0);

		// simpleThreshold(input, 50);

		// output image of same dimensions
		// final Dimensions dim = input;
		// final Img< UnsignedByteType > output = factory.create( dim );

		// process input - result in output
		// simpleThreshold( input, output, 50 );
	

		// save output image
		// Clear the file if it already exists.
		final String outPath = args[1];
		File path = new File(outPath);
		if (path.exists()) {
			path.delete();
		}
		ImgSaver imgSaver = new ImgSaver();
		imgSaver.saveImg(outPath, input);
		System.out.println("Image saved in: " + outPath);
	}

}