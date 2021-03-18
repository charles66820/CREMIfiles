package fr.ubordeaux.deptinfo.compilation.lea.parser;

class Position {
  private int line = 1;
  private int column = 1;

  public Position() {
    line = 1;
    column = 1;
  }

  public Position(int l, int t) {
    line = l;
    column = t;
  }

  public Position(Position p) {
    line = p.line;
    column = p.column;
  }

  public boolean equals(Position l) {
    return l.line == line && l.column == column;
  }

  public String toString() {
    return Integer.toString(line) + "." + Integer.toString(column);
  }

  public int getLine() {
    return line;
  }

  public int getColumn() {
    return column;
  }
}
