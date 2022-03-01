package utile;

public class HtmlStatementBuilder extends AbstractStatementBuilder {
    public String getResult() {
        return "<html><body>\n" +
                super.statement("<br>", "&nbsp;&nbsp;") +
                "</body></html>";
    }
}
