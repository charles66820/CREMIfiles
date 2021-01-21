package exemple;

import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import io.scif.img.ImgSaver;
import java.io.File;

public class CreateGrayLevelImage {
	public static Img<UnsignedByteType> createGLImage(int w, int h) {
		final int[] dimensions = new int[] { w, h };
		final ArrayImgFactory<UnsignedByteType> factory = new ArrayImgFactory<>(new UnsignedByteType());
		final Img<UnsignedByteType> img = factory.create(dimensions);
		return img;
	}

	public static void fillGrayLevelImageRandomAccess(Img<UnsignedByteType> img) {

		final RandomAccess<UnsignedByteType> r = img.randomAccess();

		final int iw = (int) img.max(0);
		final int ih = (int) img.max(1);

		for (int x = 0; x <= iw; ++x) {

			final byte gl = (byte) (x * 255 / iw);
			for (int y = 0; y <= ih; ++y) {
				r.setPosition(x, 0);
				r.setPosition(y, 1);
				final UnsignedByteType t = r.get();
				t.set(gl);
			}
		}
	}

	public static void fillGrayLevelImageCursor(Img<UnsignedByteType> img) {

		final Cursor<UnsignedByteType> cursor = img.cursor();

		while (cursor.hasNext()) {
			cursor.fwd();
			cursor.get().set(128);
		}
	}

	public static void fillGrayLevelImageLocalizingCursor(Img<UnsignedByteType> img) {

		final Cursor<UnsignedByteType> cursor = img.localizingCursor();

		while (cursor.hasNext()) {
			cursor.fwd();
			int x = (int) cursor.getDoublePosition(0);
			int y = (int) cursor.getDoublePosition(1);
			cursor.get().set(x % 255);
		}
	}

	public static void main(final String[] args) {
		if (args.length < 1) {
			System.out.println("missing output image filename");
			System.exit(-1);
		} 
		final Img<UnsignedByteType> img = createGLImage(400, 300);
		//fillGrayLevelImageRandomAccess(img);
		//fillGrayLevelImageCursor(img);
		//fillGrayLevelImageLocalizingCursor(img);

		// save output image
		final String outPath = args[0];
		File path = new File(outPath);
		if (path.exists()) {
			path.delete();
		}
		ImgSaver imgSaver = new ImgSaver();
		imgSaver.saveImg(outPath, img);
		System.out.println("Image saved: " + outPath);
	}

}