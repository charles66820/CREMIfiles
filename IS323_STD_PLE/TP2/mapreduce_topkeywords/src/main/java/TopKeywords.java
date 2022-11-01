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
import java.util.List;
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

            String year = listMatches.get(3);
            String month = listMatches.get(10);
            String keywords = listMatches.get(8);

            //TODO: check the the whole period or the decade

            if (keywords.equals("")) return;
            String[] keywordsTab = keywords.split(";");
            if (keywordsTab.length == 0) return;

            for (String keyword : keywordsTab)
                context.write(new Text(keyword), new IntWritable(1));
        }
    }

    public static class TopKeywordsReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int totalNbInPapers = 0;
            for (IntWritable value : values) {
                int nb = value.get();
                totalNbInPapers += nb;
            }
            context.write(key, new IntWritable(totalNbInPapers));
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
        job.setOutputValueClass(IntWritable.class);

        job.setOutputFormatClass(TextOutputFormat.class);
        job.setInputFormatClass(TextInputFormat.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
