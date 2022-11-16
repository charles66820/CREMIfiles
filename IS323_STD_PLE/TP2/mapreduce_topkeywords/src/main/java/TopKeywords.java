import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class TopKeywords {
    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: yarn jar topkeywords-0.0.1.jar <dataCsvFile> [<decadeTopInputFolder>] <decadeTopOutputFolder> <keywordTopOutputFolder>");
            System.exit(128);
        }
        String dataCsvFile = args[0];
        String decadeTopInputFolder;
        String decadeTopOutputFolder;
        String keywordTopOutputFolder;
        if (args.length == 3) {
            decadeTopOutputFolder = args[1];
            keywordTopOutputFolder = args[2];
        } else {
            decadeTopInputFolder = args[1];
            decadeTopOutputFolder = args[2];
            keywordTopOutputFolder = args[3];
        }

        Configuration decadeConf = new Configuration();
        Job decadeJob = Job.getInstance(decadeConf, "TopDecade");
        String nbReduceTasks = System.getenv("NB_REDUCE_TASKS");
        if (nbReduceTasks != null) decadeJob.setNumReduceTasks(Integer.parseInt(nbReduceTasks));
        decadeJob.setJarByClass(TopKeywords.class);

        FileInputFormat.addInputPath(decadeJob, new Path(dataCsvFile));
        decadeJob.setInputFormatClass(TextInputFormat.class);

        decadeJob.setMapperClass(RawDataMapper.class);
        decadeJob.setMapOutputKeyClass(IntWritable.class);
        decadeJob.setMapOutputValueClass(Text.class);

//        decadeJob.setCombinerClass(DecadeCombiner.class);

        decadeJob.setReducerClass(DecadeReducer.class);
        decadeJob.setOutputKeyClass(NullWritable.class);
        decadeJob.setOutputValueClass(Text.class);

        FileOutputFormat.setOutputPath(decadeJob, new Path(decadeTopOutputFolder));
        decadeJob.setOutputFormatClass(TextOutputFormat.class);

        if (!decadeJob.waitForCompletion(true))
            System.exit(1);

        Configuration keywordConf = new Configuration();
        Job keywordJob = Job.getInstance(keywordConf, "TopKeyword");
        keywordJob.setNumReduceTasks(1);
        keywordJob.setJarByClass(TopKeywords.class);

        FileInputFormat.addInputPath(keywordJob, new Path(decadeTopOutputFolder));
        keywordJob.setInputFormatClass(TextInputFormat.class);

        keywordJob.setMapperClass(DecadeMapper.class);
        keywordJob.setMapOutputKeyClass(Text.class);
        keywordJob.setMapOutputValueClass(Text.class);

//        keywordJob.setCombinerClass(DecadeCombiner.class);

        keywordJob.setReducerClass(KeywordReducer.class);
        keywordJob.setOutputKeyClass(NullWritable.class);
        keywordJob.setOutputValueClass(Text.class);

        keywordJob.setOutputFormatClass(TextOutputFormat.class);
        keywordJob.setInputFormatClass(TextInputFormat.class);

        FileOutputFormat.setOutputPath(keywordJob, new Path(keywordTopOutputFolder));
        keywordJob.setOutputFormatClass(TextOutputFormat.class);

        System.exit(keywordJob.waitForCompletion(true) ? 0 : 1);
    }
}
