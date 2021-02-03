public enum Sym {
    KEYWORDS("Keywords"),
    IDENTIFIER("Identifier"),
    INTEGER("Integer"),
    FLOAT("Float"),
    OPERATORS("Operators"),
    SEPATATORS("Separators"),
    STRINGS("Strings");

    private final String name;

    Sym(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return name;
    }
}
