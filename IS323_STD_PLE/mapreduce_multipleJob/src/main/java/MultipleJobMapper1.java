import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class MultipleJobMapper1 extends Mapper<Object, Text, Text, Text> {
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        if (value.toString().equals("\"author\",\"booktitle\",\"title\",\"year\",\"volume\",\"number\",\"pages\",\"abstract\",\"keywords\",\"doi\",\"month\",\"journal\",\"issn\",\"publisher\",\"isbn\",\"url\",\"order\",\"digital-lib\""))
            return;
        List<String> listMatches = new ArrayList<>();
        Pattern CSV_PATTERN = Pattern.compile("(?:,|\n|^)(\"(?:(?:\"\")*[^\"]*)*\"|[^\",\n]*|(?:\n|$))");
        Matcher m = CSV_PATTERN.matcher(value.toString());
        while (m.find()) listMatches.add(m.group(1));
        String keywords = listMatches.get(8);
        if (keywords == null || keywords.equals("")) return;
        keywords = keywords.replace("\"", "");
        if (keywords.equals("")) return;
        String[] keywordsTab = keywords.split(";");
        if (keywordsTab.length == 0) return;
        for (String keyword : keywordsTab)
            context.write(new Text(keyword + "_M1"), new Text(""));
    }
}
