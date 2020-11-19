package lessons;

public class InstrumentLesson extends Lesson {
    private final String instrument;

    public InstrumentLesson(int i, int i1, String prof1, String saxophone) {
        super(i, i1, prof1, 1);
        this.instrument = saxophone;
        this.setNbRegistrations(1);
    }

    @Override
    public String toString() {
        return "Instrument Lesson " +
                super.toString() +
                "[instrument=" + instrument +
                ']';
    }
}