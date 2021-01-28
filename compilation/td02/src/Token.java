class Token {
    private Sym sym;
    private Object value;
    private Integer lineno;
    private Integer colno;

    public Token(Sym sym, Object value, int lineno, int colno) {
        this.sym = sym;
        this.value = value;
        this.lineno = lineno;
        this.colno = colno;
    }

    @Override
    public String toString() {
        return "Token{" +
                "sym=" + sym +
                ", value=" + value +
                ", lineno=" + lineno +
                ", colno=" + colno +
                '}';
    }
}