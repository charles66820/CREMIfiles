from pyspark import SparkContext, SparkConf

#a simple function to remove some punctuation from a string
def remove_symbols(word):
	new = word.replace("?","").replace("!","").replace(",","").replace(".","").replace("<","").replace(">","").replace(":","").replace(";","")
	return new

#----------------------------------------------------------------------
#initialize Spark
conf = SparkConf().setAppName("Miserables_python")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")  #to hide the INFO messages that are usually
			#shown in the console during the execution
#read the file, which is stored in HDFS, into an RDD. Each element of 
#the RDD is a string (a line of the file). Then the map transformation 
#will apply a function (passed as argument) to each of these elements
#to generate exactly one element of a new RDD.
lines=sc.textFile("/data/LesMiserables.txt").map(remove_symbols)
print("Number of lines:")
print(lines.count())
#break each element of the RDD into a list of strings (the words). 
#flatMap means each element will become multiple elements of the 
#resulting RDD. words is therefore an RDD where each element is a word 
#of the file (a string). Notice that instead of providing a function to
#map/flatMap we can use lambda expressions.
words = lines.flatMap(lambda x: x.split())
print(words.take(5))
print("Number of words:")
print(words.count())
#Use map to transform each element of the words RDD into a pair 
#(word, 1), then use reduceByKey in this new RDD to do a reduce 
#operation. All values associated with the same key will be combined, 
#two at a time, using the provided function (in this case, a sum). The
#resulting RDD has one element per word, each of them a pair 
#(word, number of occurrences). Then do a final map transformation to
#invert each element into (number of occurrences, word).
count = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x+y).map(lambda x: (x[1], x[0]))
print("Unique words:")
print(count.count())
print(count.take(10))
#The goal of having the count as key and the word as value is being 
#able to obtain the 10 most frequently used words with the top() 
#action.
print("Top 10:")
print(count.top(10))

