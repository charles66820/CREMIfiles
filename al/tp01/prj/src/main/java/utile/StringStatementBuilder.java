package utile;

public class StringStatementBuilder extends AbstractStatementBuilder {
    public String getResult() {
        return statement("\n", "\t");
    }
}
