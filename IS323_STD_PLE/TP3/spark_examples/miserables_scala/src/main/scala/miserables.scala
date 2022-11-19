import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD

object Miserables_Scala {
	//a simple function to remove some punctuation from a string
	def remove_symbols(word : String): String = {
		return word.replaceAll("""[\p{Punct}]""", "")
	}
	//execution starts by the main function
	def main(args: Array[String]): Unit = {
		//initialize Spark (the two lines below are not required when using the interactive shell, where we can simply use sc directly)
		val conf = new SparkConf().setAppName("Miserables_scala")
		val sc = new SparkContext(conf)
		//hide the INFO messages that are usually shown in the console during the execution
		sc.setLogLevel("ERROR")
		//read the file, which is stored in HDFS, into an RDD. Each element of the RDD is a string (a line of the file). Then the map transformation will apply a function (passed as argument) to each of these elements to generate exactly one element of a new RDD.
		val lines = sc.textFile("/data/LesMiserables.txt").map(remove_symbols)
		println("Number of lines:")
		println(lines.count())
		//break each element of the RDD into a list of strings (the words). flatMap means each element will become multiple elements of the resulting RDD. words is therefore an RDD where each element is a word of the file (a string). Finally, we filter out the empty word "". Notice that instead of providing a function to map/flatMap/filter we can use lambda expressions
		val words = lines.flatMap(x => x.split("\\s+")).filter(x => x.length > 0)
		words.take(5).foreach(println) //if we call println(words.take(5)), it will show something like "[Ljava.lang.String;@3a6dd085", it does not know how to directly print an Array
		println("Number of words:")
		println(words.count())
		//use map to transform each element of the words RDD into a pair (word, 1), then use reduceByKey in this new RDD to do a reduce operation. All values associated with the same key will be combined, two at a time, using the provided function (in this case, a sum). The resulting RDD has one element per word, each of them a pair (word, numbre of occurrences). Then do a final map transformation to reverse each element into (number of occurrences, word).
		val count =  words.map((_,1)).reduceByKey(_+_).map(x => (x._2, x._1))  //in scala, sometimes we can simplify the lambda expression by using _ instead of giving a name (like "x=>")
		println("Unique words:")
		println(count.count())
		count.take(10).foreach(println)
		//The goal of having the count as key and the word as value is being able to obtain the 10 most frequently used words with the top() action.
		println("Top 10:")
		count.top(10).foreach(println)
	}
}
