import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TopKeywords {
    public static class TopKeywordsMapper extends Mapper<Object, Text, Text, Text> {
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

//            Counter minDecadeCounter = context.getCounter("decade", "min");
//            long minDecadeCounterVal = minDecadeCounter.getValue();
//            if (minDecadeCounterVal > decade)
//                minDecadeCounter.setValue(decade);
//
//            Counter maxDecadeCounter = context.getCounter("decade", "max");
//            long maxDecadeCounterVal = maxDecadeCounter.getValue();
//            if (maxDecadeCounterVal < decade)
//                maxDecadeCounter.setValue(decade);

            keywords = keywords.replace("\"", "");

            if (keywords.equals("")) return;
            String[] keywordsTab = keywords.split(";");
            if (keywordsTab.length == 0) return;

            for (String keyword : keywordsTab)
                context.write(new Text(String.valueOf(decade)), new Text(keyword));
        }
    }

    public static class DecadeCombiner extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text decade, Iterable<Text> keywords, Context context) throws IOException, InterruptedException {
            // Count number of paper for each keyword
            Map<String, Integer> keywordsCount = new HashMap<>();
            for (Text value : keywords) {
                String keyword = value.toString();
                keywordsCount.merge(keyword, 1, Integer::sum);
            }

            // Top keywords by decades
            final TreeMap<Integer, String> topKeywords = new TreeMap<>(
                    Comparator.reverseOrder());
            for (Map.Entry<String, Integer> entry : keywordsCount.entrySet())
                topKeywords.put(entry.getValue(), entry.getKey());

            int decadeTop = 1;
            for (Map.Entry<Integer, String> entry : topKeywords.entrySet()) {
                Integer nbPaperInDecade = entry.getKey();
                String keyword = entry.getValue();
                // keyword : decade ; nbPaperInDecade ; decadeTop
                context.write(new Text(keyword), new Text(decade + ";" + nbPaperInDecade + ";" + decadeTop));
                decadeTop += 1;
            }
        }
    }

    public static class DecadeReducer extends Reducer<Text, Text, NullWritable, MapWritable> {
        private final Map<Integer, MapWritable> topKeywords = new TreeMap<>(
                Comparator.reverseOrder());

        public void reduce(Text keyword, Iterable<Text> values, Context context) throws InterruptedException {
            MapWritable data = new MapWritable();
            data.put(new Text("keyword"), new Text(keyword.toString()));

            int totalNbInPaper = 0;
            for (Text value : values) {
                String[] keywordDecateInfo = value.toString().split(";");
                String decade = keywordDecateInfo[0];
                String nbPaperInDecade = keywordDecateInfo[1];
                String decadeTop = keywordDecateInfo[2];
                data.put(new Text("nbDecade" + decade), new Text(nbPaperInDecade));
                data.put(new Text("topDecade" + decade), new Text(decadeTop));
                totalNbInPaper += Integer.parseInt(nbPaperInDecade);
            }

            data.put(new Text("totalNbInPaper"), new IntWritable(totalNbInPaper));
            topKeywords.put(totalNbInPaper, data);
        }

        public void cleanup(Context context) throws IOException, InterruptedException {
            int decadeTop = 1;
            for (Map.Entry<Integer, MapWritable> entry : topKeywords.entrySet()) {
                MapWritable data = entry.getValue();
                data.put(new Text("top"), new IntWritable(decadeTop));
                context.write(NullWritable.get(), data);
                decadeTop += 1;
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();

        Job job = Job.getInstance(conf, "TopKeywords");
        job.setNumReduceTasks(1);
        job.setJarByClass(TopKeywords.class);

        job.setMapperClass(TopKeywordsMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setCombinerClass(DecadeCombiner.class);

        job.setReducerClass(DecadeReducer.class);
//        job.setOutputKeyClass(IntWritable.class);
//        job.setOutputValueClass(Text.class);

        job.setOutputFormatClass(TextOutputFormat.class);
        job.setInputFormatClass(TextInputFormat.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
