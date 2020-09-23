class tutu {
    private int a;
    private int b;
    tutu(int a, int b) {
        this.a = a;
        this.b = b;
    }
    public void test(tutu t) {
        this.a = t.a;
        this.b = t.b;
    }
    public void print() {
        System.out.println("tutu a : " + this.a + " b : " + this.b);
    }
}
class test {
    public static void main(String[] args) {
        tutu t = new tutu(2, 6);
        tutu t2 = new tutu(5, 7);
        t.print();
        t2.print();
        t.test(t2);
        t.print();
    }
}