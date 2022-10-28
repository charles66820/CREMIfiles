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
import java.util.Comparator;
import java.util.Map;
import java.util.TreeMap;

public class TopCity {
    public static class TopCityMapper extends Mapper<Object, Text, IntWritable, Text> {
        private final Map<Integer, String> sm = new TreeMap<Integer, String>(
                new Comparator<Integer>() {
                    @Override
                    public int compare(Integer k1, Integer k2) {
                        return k2.compareTo(k1);
                    }
                });

        public void cleanup(Context context) throws IOException, InterruptedException {
            int k = Integer.parseInt(context.getConfiguration().get("K"));

            ArrayList<String> values = new ArrayList<String>(sm.values());
            for (int i = 0; i < Math.min(k, values.size()); i++) {
                String v = values.get(i);
                String[] popCity = v.split("\t");

                int population = Integer.parseInt(popCity[0]);
                String city = popCity[1];

                IntWritable popMag = new IntWritable((int) Math.pow(10, (int) Math.log10(population)));
                context.write(popMag, new Text(population + "\t" + city));
            }
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            if (value.toString().equals("Country,City,AccentCity,Region,Population,Latitude,Longitude")) return;
            String[] rowTab = value.toString().split(",");
            String city = rowTab[1];

            if (rowTab[4] == null || rowTab[4].equals("")) return;
            int population = Integer.parseInt(rowTab[4]);

            sm.put(population, population + "\t" + city);
        }
    }

    public static class TopCityReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
        private final Map<Integer, String> sm = new TreeMap<Integer, String>(
                new Comparator<Integer>() {
                    @Override
                    public int compare(Integer k1, Integer k2) {
                        return k2.compareTo(k1);
                    }
                });

        public void cleanup(Context context) throws IOException, InterruptedException {
            int k = Integer.parseInt(context.getConfiguration().get("K"));

            ArrayList<String> values = new ArrayList<String>(sm.values());
            for (int i = 0; i < Math.min(k, values.size()); i++) {
                String v = values.get(i);
                String[] popCity = v.split("\t");

                int population = Integer.parseInt(popCity[0]);
                String city = popCity[1];

                context.write(new IntWritable(0), new Text(population + "\t" + city));
            }
        }

        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            for (Text value : values) {
                String[] popCity = value.toString().split("\t");
                int population = Integer.parseInt(popCity[0]);
                sm.put(population, value.toString());
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        int k = args.length >= 3 ? Integer.parseInt(args[2]) : 10;
        conf.set("K", String.valueOf(k));

        Job job = Job.getInstance(conf, "TopCity");
        job.setNumReduceTasks(1);
        job.setJarByClass(TopCity.class);

        job.setMapperClass(TopCityMapper.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);

        job.setCombinerClass(TopCityReducer.class);

        job.setReducerClass(TopCityReducer.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);

        job.setOutputFormatClass(TextOutputFormat.class);
        job.setInputFormatClass(TextInputFormat.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
