package school;

import lessons.Lesson;

public class MusicSchool {
    private final Lesson[] lessons;
    private int lesonsCount;

    public MusicSchool() {
        this.lessons = new Lesson[10];
    }

    public void addLesson(Lesson i1) throws MaxSizeExceededException {
        if (lesonsCount >= 10) throw new MaxSizeExceededException("error");
        this.lessons[lesonsCount] = i1;
        this.lesonsCount++;
    }

    public int getBalance() {
        int s = 0;
        for (int i = 0; i < this.lesonsCount; i++) {
            s += this.lessons[i].getBalance();
        }
        return s;
    }

    @Override
    public String toString() {
        StringBuilder r = new StringBuilder("MusicSchool\n");
        for (int i = 0; i < this.lesonsCount; i++) {
            r.append(this.lessons[i].toString()).append("\n");
        }
        return r.toString();
    }
}