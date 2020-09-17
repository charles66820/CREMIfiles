public class Point {
    private double x;
    private double y;

    public Point() {}

    public Point(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public double getX() {
        return this.x;
    }

    public void setX(double v) {
        this.x = v;
    }

    public double getY() {
        return this.y;
    }

    public void setY(double v) {
        this.y = v;
    }

    public void afficher() {
        System.out.println("x: " + this.x + " y: " + this.y);
    }

    public void deplacement(double dx, double dy) {
        this.x += dx;
        this.y += dy;
    }

    public void deplacement(double delta) {
        this.x += delta;
        this.y += delta;
    }

    public double distance(Point p) {
        return Math.sqrt(Math.pow(p.getX() - this.x, 2) + Math.pow(p.getY() - this.y, 2));
    }

}