package lessons;

public abstract class Lesson {
    private final int annualCost;
    private final int price;
    private final String professor;
    private int nbRegistrations;
    private final int nbRegistrationsMax;

    public Lesson(int annualCost, int price, String professor, int nbRegistrationsMax) {
        this.annualCost = annualCost;
        this.price = price;
        this.professor = professor;
        this.nbRegistrationsMax = nbRegistrationsMax;
    }

    public void setNbRegistrations(int nbRegistrations) {
        this.nbRegistrations = nbRegistrations;
    }

    public int getNbRegistrations() {
        return this.nbRegistrations;
    }

    public int getNbRegistrationsMax() {
        return nbRegistrationsMax;
    }

    public int getBalance() {
        return (this.price * this.nbRegistrations) - this.annualCost;
    }

    @Override
    public String toString() {
        return "[" +
                "annualCost=" + annualCost +
                ", price=" + price +
                ", professor=" + professor +
                ", registrations=" + nbRegistrations +
                ']';
    }
}