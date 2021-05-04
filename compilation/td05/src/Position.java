class Position {
    final private int line;
    final private int column;

    public Position(int line, int column) {
        this.line = line;
        this.column = column;
    }

    // format d'affichage d'une position
    public String toString() {
        return String.format("[ %d : %d ]", line, column);
    }
}