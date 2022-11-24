import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class DecadeReducer1 extends Reducer<Text, Text, Text, Text> {
    public void reduce(Text keyword, Iterable<Text> values, Context context) throws InterruptedException, IOException {
        context.write(new Text(keyword.toString() + "_R1"), new Text(""));
    }
}
