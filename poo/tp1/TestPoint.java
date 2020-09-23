class TestPoint {
    public static void main(String[] args) {
        Point p = new Point(4.0, 3.0);
        p.print();
        p.move(2.0, 6.0);
        p.print();
        p.move(-2.0);
        p.print();
        Point p2 = new Point(6.0, 29.0);
        System.out.println(p.distance(p2));
    }
}