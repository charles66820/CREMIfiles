class Triangle {
    private Point2D p1;
    private Point2D p2;
    private Point2D p3;

    int r;
    int g;
    int b;

    public Triangle(Point2D p1, Point2D p2, Point2D p3) {
      this.p1 = p1;
      this.p2 = p2;
      this.p3 = p3;
    }

    public int getR() {
        return this.r;
    }

    public void setR(int v) {
        this.r = v;
    }

    public int getG() {
        return this.g;
    }

    public void setG(int v) {
        this.g = v;
    }

    public int getB() {
        return this.b;
    }

    public void setB(int v) {
        this.b = v;
    }

    public double perimeter() {
        return this.p1.distance(this.p2) + this.p2.distance(this.p3) + this.p3.distance(this.p1);
    }

    public boolean isIsosceles() {
      return this.p1.distance(this.p2) == this.p2.distance(this.p3) || this.p1.distance(this.p2) == this.p3.distance(this.p1);
    }


    public String svg() {
        return "<line x1='" + this.p1.getX() + "' y1='" + this.p1.getY() + "' x2='" + this.p2.getX() + "' y2='" + this.p2.getY() + "' stroke='rgb(" + this.r + "," + this.g + "," + this.b + ")' stroke-width='3' />" +
        "<line x1='" + this.p2.getX() + "' y1='" + this.p2.getY() + "' x2='" + this.p3.getX() + "' y2='" + this.p3.getY() + "' stroke='rgb(" + this.r + "," + this.g + "," + this.b + ")' stroke-width='3' />" +
        "<line x1='" + this.p3.getX() + "' y1='" + this.p3.getY() + "' x2='" + this.p1.getX() + "' y2='" + this.p1.getY() + "' stroke='rgb(" + this.r + "," + this.g + "," + this.b + ")' stroke-width='3' />";
    }
}