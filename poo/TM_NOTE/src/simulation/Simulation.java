package simulation;

import lessons.*;
import school.MaxSizeExceededException;
import school.MusicSchool;


public class Simulation {
    public static void main(String args[]) throws MaxSizeExceededException {

        InstrumentLesson i1 = new InstrumentLesson(1200, 670, "Prof1", "Saxophone");
        InstrumentLesson i2 = new InstrumentLesson(1000, 500, "Prof2", "Piano");
        TheoryLesson t1 = new TheoryLesson(1300, 215, "Prof3", 12);
        TheoryLesson t2 = new TheoryLesson(1200, 215, "Prof3", 10);
        t1.addRegistration(7);
        MusicSchool school = new MusicSchool();
        school.addLesson(i1);
        school.addLesson(i2);
        school.addLesson(t1);
        school.addLesson(t2);

        System.out.print(school);
        System.out.println("Balance : " + school.getBalance());

        for (int i = 0; i < 12; ++i) {
            int added = t2.addRegistration();
            if (added == 1) {
                System.out.println("One more registration. Total = " + t2.getNbRegistrations());
                System.out.println("Balance : " + school.getBalance());
            }
        }
    }
}
