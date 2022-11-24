import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ExistingDataDecadeMapper extends Mapper<Object, Text, IntWritable, Text> {
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        if (value.toString().equals("decade;keyword;nbPaperInDecade;decadeTop")) return;

        String[] rowTab = value.toString().split(";");
        if(rowTab.length < 4) return;

        String decade = rowTab[0];
        String keyword = rowTab[1];
        String nbPaperInDecade = rowTab[2];
        String decadeTop = rowTab[3];

        if (decade == null || decade.equals("") || keyword == null || keyword.equals("") || nbPaperInDecade == null || nbPaperInDecade.equals("") || decadeTop == null || decadeTop.equals(""))
            return;

        int nbPaperInDecadeVal;
        try {
            nbPaperInDecadeVal = Integer.parseInt(nbPaperInDecade);
        } catch (NumberFormatException ex) {
            return;
        }

        context.write(new IntWritable(Integer.parseInt(decade)), new Text(keyword + ";" + nbPaperInDecade + ";" + decadeTop));
    }
}
