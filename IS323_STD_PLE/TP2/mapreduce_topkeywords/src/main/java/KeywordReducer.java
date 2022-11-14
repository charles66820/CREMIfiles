import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.*;

public class KeywordReducer extends Reducer<Text, Text, NullWritable, Text> {
    private final Map<Integer, Map<String, String>> topKeywords = new TreeMap<>(
            Comparator.reverseOrder());
    SortedSet<Integer> decateSet = new TreeSet<>();

    public void reduce(Text keyword, Iterable<Text> values, Context context) {
        Map<String, String> data = new HashMap<>();
        data.put("keyword", keyword.toString());

        int totalNbInPaper = 0;
        for (Text value : values) {
            String[] keywordDecateInfo = value.toString().split(";");
            String decade = keywordDecateInfo[0];
            String nbPaperInDecade = keywordDecateInfo[1];
            String decadeTop = keywordDecateInfo[2];
            data.put("nbDecade" + decade, nbPaperInDecade);
            data.put("topDecade" + decade, decadeTop);
            decateSet.add(Integer.valueOf(decade));
            totalNbInPaper += Integer.parseInt(nbPaperInDecade);
        }

        topKeywords.put(totalNbInPaper, data);
    }

    public void cleanup(Context context) throws IOException, InterruptedException {
        StringBuilder header = new StringBuilder("keyword;top;totalNbInPaper");
        for (Integer decade : decateSet)
            header.append(";topDecade").append(decade).append(";nbDecade").append(decade);
        context.write(NullWritable.get(), new Text(header.toString()));

        int top = 1;
        for (Map.Entry<Integer, Map<String, String>> entry : topKeywords.entrySet()) {
            Map<String, String> data = entry.getValue();

            StringBuilder keywordRow = new StringBuilder();
            keywordRow.append(data.get("keyword"));
            keywordRow.append(top);
            keywordRow.append(data.get("totalNbInPaper"));

            for (Integer decade : decateSet) {
                keywordRow.append(data.get("topDecade" + decade));
                keywordRow.append(data.get("nbDecade" + decade));
            }

            context.write(NullWritable.get(), new Text(keywordRow.toString()));
            top += 1;
        }
    }
}
