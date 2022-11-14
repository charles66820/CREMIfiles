import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class MultipleJob {
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job1 = Job.getInstance(conf, "rawProcess");
        job1.setNumReduceTasks(4);
        job1.setJarByClass(MultipleJob.class);

        FileInputFormat.addInputPath(job1, new Path(args[0]));
        job1.setInputFormatClass(TextInputFormat.class);

        job1.setMapperClass(MultipleJobMapper1.class);
        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(Text.class);

        job1.setReducerClass(DecadeReducer1.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);

        FileOutputFormat.setOutputPath(job1, new Path("tmpFile"));
        job1.setOutputFormatClass(TextOutputFormat.class);

        job1.waitForCompletion(true);

        Configuration conf2 = new Configuration();
        Job job2 = Job.getInstance(conf2, "secondProcess");
        job2.setNumReduceTasks(1);
        job2.setJarByClass(MultipleJob.class);

        FileInputFormat.addInputPath(job2, new Path("tmpFile"));
        job2.setInputFormatClass(TextInputFormat.class);

        job2.setMapperClass(MultipleJobMapper2.class);
        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(Text.class);

        job2.setReducerClass(DecadeReducer2.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);

        FileOutputFormat.setOutputPath(job2, new Path(args[1]));
        job2.setOutputFormatClass(TextOutputFormat.class);

        System.exit(job2.waitForCompletion(true) ? 0 : 1);
    }
}
