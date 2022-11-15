import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class RawDataMapper extends Mapper<Object, Text, IntWritable, Text> {
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        if (value.toString().equals("\"author\",\"booktitle\",\"title\",\"year\",\"volume\",\"number\",\"pages\",\"abstract\",\"keywords\",\"doi\",\"month\",\"journal\",\"issn\",\"publisher\",\"isbn\",\"url\",\"order\",\"digital-lib\""))
            return;

        // String[] rowTab = value.toString().split(","); // BUG: don't support csv escape char
        // Regex csv split found at https://gist.github.com/awwsmm/886ac0ce0cef517ad7092915f708175f
        List<String> listMatches = new ArrayList<>();
        Pattern CSV_PATTERN = Pattern.compile("(?:,|\n|^)(\"(?:(?:\"\")*[^\"]*)*\"|[^\",\n]*|(?:\n|$))");
        Matcher m = CSV_PATTERN.matcher(value.toString());
        while (m.find()) listMatches.add(m.group(1));

        String rawYear = listMatches.get(3);
        String keywords = listMatches.get(8);

        // Ignore undefined year and undefined keywords.
        if (rawYear == null || rawYear.equals("") || keywords == null || keywords.equals(""))
            return;

        // Clean data
        rawYear = rawYear.replaceAll("\"", "");

        int year;
        try {
            year = Integer.parseInt(rawYear);
        } catch (NumberFormatException ex) {
            return;
        }

        int decade = year / 10;

        keywords = keywords.replace("\"", "");

        if (keywords.equals("")) return;
        String[] keywordsTab = keywords.split(";");
        if (keywordsTab.length == 0) return;

        for (String keyword : keywordsTab)
            context.write(new IntWritable(decade), new Text(keyword));
    }
}
