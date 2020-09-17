class Cercle {
    private double rayon;
    private Point centre;

    int r;
    int g;
    int b;

    public Cercle(Point centre, double rayon) {
        this.centre = centre;
        this.rayon = rayon;
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

    // retourne la surface du cercle (pi fois le carré du rayon)
    public double surface() {
        return Math.PI * Math.sqrt(this.rayon);
    }

    // teste si le point p passé en paramètre fait ou non partie du cercle (frontière comprise : disque fermé). La méthode retournera `true` si le test est positif, et `false` dans le cas contraire.
    public boolean estInterieur(Point p) {
        return p.distance(this.centre) <= rayon;
    }

    public String svg() {
        return "<circle cx='" + this.centre.getX() + "' cy='" + this.centre.getY() + "' r='" + this.rayon + "' fill='rgb(" + this.r + "," + this.g + "," + this.b + ")' />";
    }
}