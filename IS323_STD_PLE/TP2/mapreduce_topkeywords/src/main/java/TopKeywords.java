import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class TopKeywords {
    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: " + args[0] + " <dataCsvFile> [<decadeTopInputFolder>] <decadeTopOutputFolder> <keywordTopOutputFolder>");
            System.exit(128);
        }
        String dataCsvFile = args[1];
        String decadeTopInputFolder;
        String decadeTopOutputFolder;
        String keywordTopOutputFolder;
        if (args.length == 4) {
            decadeTopOutputFolder = args[2];
            keywordTopOutputFolder = args[3];
        } else {
            decadeTopInputFolder = args[2];
            decadeTopOutputFolder = args[3];
            keywordTopOutputFolder = args[4];
        }


        Configuration decadeConf = new Configuration();
        Job decadeJob = Job.getInstance(decadeConf, "TopDecade");
        decadeJob.setJarByClass(TopKeywords.class);

        FileInputFormat.addInputPath(decadeJob, new Path(dataCsvFile));
        decadeJob.setInputFormatClass(TextInputFormat.class);

        decadeJob.setMapperClass(RawDataMapper.class);
        decadeJob.setMapOutputKeyClass(Text.class);
        decadeJob.setMapOutputValueClass(Text.class);

//        decadeJob.setCombinerClass(DecadeCombiner.class);

        decadeJob.setReducerClass(DecadeReducer.class);
        decadeJob.setOutputKeyClass(Text.class);
        decadeJob.setOutputValueClass(Text.class);

        FileOutputFormat.setOutputPath(decadeJob, new Path(decadeTopOutputFolder));
        decadeJob.setOutputFormatClass(TextOutputFormat.class);

        if (!decadeJob.waitForCompletion(true))
            System.exit(1);

        Configuration keyWordConf = new Configuration();
        Job keyWordJob = Job.getInstance(keyWordConf, "TopKeyWord");
        keyWordJob.setNumReduceTasks(1);
        keyWordJob.setJarByClass(TopKeywords.class);

        FileInputFormat.addInputPath(keyWordJob, new Path(decadeTopOutputFolder));
        keyWordJob.setInputFormatClass(TextInputFormat.class);

        keyWordJob.setMapperClass(DecadeMapper.class);
        keyWordJob.setMapOutputKeyClass(Text.class);
        keyWordJob.setMapOutputValueClass(Text.class);

//        keyWordJob.setCombinerClass(DecadeCombiner.class);

        keyWordJob.setReducerClass(KeywordReducer.class);
        keyWordJob.setOutputKeyClass(NullWritable.class);
        keyWordJob.setOutputValueClass(Text.class);

        keyWordJob.setOutputFormatClass(TextOutputFormat.class);
        keyWordJob.setInputFormatClass(TextInputFormat.class);

        FileOutputFormat.setOutputPath(keyWordJob, new Path(keywordTopOutputFolder));
        keyWordJob.setOutputFormatClass(TextOutputFormat.class);

        System.exit(keyWordJob.waitForCompletion(true) ? 0 : 1);
    }
}
