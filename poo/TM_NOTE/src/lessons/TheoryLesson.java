package lessons;

public class TheoryLesson extends Lesson {
    public TheoryLesson(int i, int i1, String prof3, int i2) {
        super(i, i1, prof3, i2);
    }

    public int addRegistration(int i) {
        int canAdd = this.getNbRegistrationsMax() - this.getNbRegistrations();
        if (i <= canAdd) {
            this.setNbRegistrations(this.getNbRegistrations() + i);
            return i;
        } else {
            this.setNbRegistrations(this.getNbRegistrations() + canAdd);
            return canAdd;
        }
    }

    public int addRegistration() {
        return addRegistration(1);
    }

    @Override
    public String toString() {
        return "Theory Lesson " + super.toString();
    }
}
