import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TopKeywords {
    public static class TopKeywordsMapper extends Mapper<Object, Text, Text, IntWritable> {
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            if (value.toString().equals("\"author\",\"booktitle\",\"title\",\"year\",\"volume\",\"number\",\"pages\",\"abstract\",\"keywords\",\"doi\",\"month\",\"journal\",\"issn\",\"publisher\",\"isbn\",\"url\",\"order\",\"digital-lib\""))
                return;

            // String[] rowTab = value.toString().split(","); // BUG: don't support csv escape char
            // Regex csv split found at https://gist.github.com/awwsmm/886ac0ce0cef517ad7092915f708175f
            List<String> listMatches = new ArrayList<String>();
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
                context.write(new Text(keyword), new IntWritable(decade));
        }
    }

    public static class TopKeywordsReducer extends Reducer<Text, IntWritable, Text, Text> {
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            final TreeMap<Integer, Integer> decates = new TreeMap<Integer, Integer>();
            int totalNbInPapers = 0;
            for (IntWritable value : values) {
                int decate = value.get();
                decates.put(decate, decates.containsKey(decate) ? decates.get(decate) + 1 : 1);
                totalNbInPapers += 1;
            }

            context.write(key, new Text(totalNbInPapers + " " + Arrays.toString(decates.values().toArray())));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("K", "10");

        Job job = Job.getInstance(conf, "TopKeywords");
        job.setNumReduceTasks(1);
        job.setJarByClass(TopKeywords.class);

        job.setMapperClass(TopKeywordsMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        // job.setCombinerClass(TopKeywordsReducer.class);

        job.setReducerClass(TopKeywordsReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        job.setOutputFormatClass(TextOutputFormat.class);
        job.setInputFormatClass(TextInputFormat.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
