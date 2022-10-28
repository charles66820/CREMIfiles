import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;
import java.lang.reflect.Array;

public class CityPop {
    public static class CityPopMapper extends Mapper<Object, Text, IntWritable, IntWritable> {
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            if (value.toString().equals("Country,City,AccentCity,Region,Population,Latitude,Longitude"))
                return;
            String[] rowTab = value.toString().split(",");
            Text city = new Text(rowTab[1]);

            if (rowTab[4] == null || rowTab[4].equals("")) return;
            int population = Integer.parseInt(rowTab[4]);

            IntWritable pop = new IntWritable((int) Math.pow(10, (int) Math.log10(population)));
            context.write(pop, new IntWritable(population));
        }
    }

    public static class CityPopReducer extends Reducer<IntWritable, IntWritable, IntWritable, MapWritable> {
        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            int count = 0;
            int max = Integer.MIN_VALUE;
            int min = Integer.MAX_VALUE;

            for (IntWritable value : values) {
                int val = value.get();
                count++;
                sum += val;
                max = Math.max(max, val);
                min = Math.min(min, val);
            }

            MapWritable ret = new MapWritable();
            ret.put(new Text("count"), new IntWritable(count));
            ret.put(new Text("avg"), new IntWritable(sum / count));
            ret.put(new Text("max"), new IntWritable(max));
            ret.put(new Text("min"), new IntWritable(min));
            context.write(key, ret);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "CityPop");
        job.setNumReduceTasks(1);
        job.setJarByClass(CityPop.class);

        job.setMapperClass(CityPopMapper.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(IntWritable.class);

        // job.setCombinerClass(CityPopReducer.class);

        job.setReducerClass(CityPopReducer.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(MapWritable.class);

        job.setOutputFormatClass(TextOutputFormat.class);
        job.setInputFormatClass(TextInputFormat.class);


        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
