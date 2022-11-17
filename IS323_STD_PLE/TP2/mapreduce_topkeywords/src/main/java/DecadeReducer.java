import com.google.common.collect.Ordering;
import com.google.common.collect.TreeMultimap;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class DecadeReducer extends Reducer<IntWritable, Text, NullWritable, Text> {
    public void setup(Context context) throws IOException, InterruptedException {
        context.write(NullWritable.get(), new Text("decade;keyword;nbPaperInDecade;decadeTop"));
    }

    public void reduce(IntWritable decade, Iterable<Text> keywords, Context context) throws IOException, InterruptedException {
        // Count number of paper for each keyword
        Map<String, Integer> keywordsCount = new HashMap<>();
//        boolean isNew = false;
        for (Text value : keywords) {
            String keyword = value.toString();
            String[] keywordsTab = keyword.split(";");
//            if (keywordsTab.length == 1) isNew = true;
            keywordsCount.merge(keywordsTab[0], 1, Integer::sum);
        }

//        if (!isNew) {
//            for (Text keyword : keywords)
//                context.write(NullWritable.get(), new Text(decade + ";" + keyword.toString()));
//            return;
//        }

        // Top keywords by decades
        final TreeMultimap<Integer, String> topKeywords = TreeMultimap.create(Ordering.natural().reverse(), Ordering.natural());
        for (Map.Entry<String, Integer> entry : keywordsCount.entrySet())
            topKeywords.put(entry.getValue(), entry.getKey());

        int decadeTop = 1;
        for (Map.Entry<Integer, String> entry : topKeywords.entries()) {
            Integer nbPaperInDecade = entry.getKey();
            String keyword = entry.getValue();
            // decade : keyword ; nbPaperInDecade ; decadeTop
            context.write(NullWritable.get(), new Text(decade + ";" + keyword + ";" + nbPaperInDecade + ";" + decadeTop));
            decadeTop += 1;
        }
    }
}