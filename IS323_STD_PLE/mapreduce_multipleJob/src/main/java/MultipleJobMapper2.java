import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class MultipleJobMapper2 extends Mapper<Object, Text, Text, Text> {
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        String[] row = value.toString().split("\t");
        if (row.length == 0) return;
        String keyword = row[0];
        context.write(new Text(keyword + "_M2") , new Text(""));
    }
}
