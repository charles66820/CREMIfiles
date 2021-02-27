public enum Sym {
    BOOL("Bool"),
    BREAK("Break"),
    CASE("Case"),
    CATCH("Catch"),
    CHAR("Char"),
    CLASS("Class"),
    CONST("Const"),
    CONTINUE("Continue"),
    DEFAULT("Default"),
    DELETE("Delete"),
    DO("Do"),
    DOUBLE("Double"),
    ELSE("Else"),
    ELSEIF("Elseif"),
    ENUM("Enum"),
    FALSE("False"),
    FLOAT("Float"),
    FOR("For"),
    FRIEND("Friend"),
    GOTO("Goto"),
    IF("If"),
    INLINE("Inline"),
    INT("Int"),
    LONG("Long"),
    NAMESPACE("Namespace"),
    NEW("New"),
    OPERATOR("Operator"),
    PRIVATE("Private"),
    PROTECTED("Protected"),
    PUBLIC("Public"),
    REGISTER("Register"),
    RETURN("Return"),
    SHORT("Short"),
    SIGNED("Signed"),
    SIZEOF("Sizeof"),
    STATIC("Static"),
    STRUCT("Struct"),
    SWITCH("Switch"),
    TEMPLATE("Template"),
    THIS("This"),
    THROW("Throw"),
    TRUE("True"),
    TRY("Try"),
    TYPEDEF("Typedef"),
    TYPEID("Typeid"),
    TYPENAME("Typename"),
    UNION("Union"),
    UNSIGNED("Unsigned"),
    USING("Using"),
    VIRTUAL("Virtual"),
    VOID("Void"),
    WHILE("While"),
    IDENTIFIER("Identifier"),
    INTEGER("Integer"),
    NUMBER("Number"),
    INCREMENT("Increment"),
    PLUS_EQ("Plus equal"),
    PLUS("Plus"),
    DECREMENT("Decrement"),
    MINUS_EQ("Minus equal"),
    MINUS("Minus"),
    MULTIPLICATION_EQ("Multiplication equal"),
    MULTIPLICATION("Multiplication"),
    DIVISION_EQ("Division equal"),
    DIVISION("Division"),
    MODULO_EQ("Modulo equal"),
    MODULO("Modulo"),
    SHL_EQ("Shl equal"),
    SHL("Shl"),
    LESS_EQ("Less equal"),
    LESS("Less"),
    SHR_EQ("Shr equal"),
    SHR("Shr"),
    OGREATER_EQ("Ogreater equal"),
    GREATER("Greater"),
    AND("And"),
    AND_EQ("And equal"),
    BiTAND("Bitand"),
    OR("Or"),
    OR_EQ("Or equal"),
    BITOR("Bitor"),
    NOT_EQ("Not equal"),
    NOT("Not"),
    XOR_EQ("Xor equal"),
    XOR("Xor"),
    EQ("Eq"),
    ASSIGNMENT("Assignment"),
    COMPL("Compl"),
    COMMA("Comma"),
    SEMICOLON("Semicolon"),
    SCOPE_RESOLUTION("Scope resolution"),
    LPAR("Lpar"),
    RPAR("Rpar"),
    LBRA("Lbra"),
    RBRA("Rbra"),
    LBRACE("Lbrace"),
    RBRACE("Rbrace"),
    QUESTION("Question"),
    COLON("Colon"),
    STRING("String"),
    DOC_AUTHOR("Doc author"),
    DOC_VERSION("Doc version"),
    DOC_PARAM("Doc param"),
    DOC_RETURN("Doc return");

    private final String name;

    Sym(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return name;
    }
}
