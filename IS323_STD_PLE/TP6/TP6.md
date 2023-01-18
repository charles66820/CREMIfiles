# TP6

Charles Goedefroit

## notes

Start interactive shell :

```bash
spark-shell --master yarn
```

Build and start spark app :

```bash
cd tweets_scala
mvn package
scp target/tweets_scala-0.0.1.jar lsd_remote:
ssh lsd_remote
spark-submit --master yarn tweets_scala-0.0.1.jar
spark-submit --num-executors 2 --executor-cores 2 --master yarn tweets_scala-0.0.1.jar
```

`/user/fzanonboito/CISD/worldcitiespop.txt`

`for i in {1..20}; do cat worldcitiespop.txt >> worldcitiespopBig.txt; done`
`for i in {1..40}; do cat worldcitiespop.txt >> worldcitiespopHuge.txt; done`
`wc -l worldcitiespopBig.txt`
`63479180 worldcitiespopBig.txt`

## Load data

```scala
val data = sc.textFile("worldcitiespopBig.txt")
val dataWithoutHeader = data.mapPartitionsWithIndex((i, r) => if (i == 0) r.drop(1) else r)
val dataStringsList = dataWithoutHeader.map(_.replace("\"", "").split(','))
dataStringsList.persist()
val cityPop = dataStringsList.map(x => (try { x(4).toInt } catch {case e: Exception => 0}, x(1)))
cityPop.take(4)foreach(println)
```

## 1. Best top k in spark

### 1. by hand (rdd.mapPartitions().top(k))

```scala
import com.google.common.collect.{TreeMultimap}
import scala.collection.JavaConverters._

cityPop.unpersist()
val tBegin = System.nanoTime
val res = cityPop.mapPartitions(x => {
  var topK = TreeMultimap.create[Int, String](Ordering[Int].reverse,Ordering[String])
  x.foreach(y => topK.put(y._1, y._2))
  topK.entries().asScala.map(y => (y.getKey(), y.getValue())).iterator
}).top(20)
println("exec in " + ((System.nanoTime - tBegin) / 1e9d))
res.foreach(println)
```

> 8.832295786 seconds
> 164.904389486 seconds

### 2. just top (rdd.top(k))

```scala
cityPop.unpersist()
val tBegin = System.nanoTime
var res = cityPop.top(20)
println("exec in " + ((System.nanoTime - tBegin) / 1e9d))
res.foreach(println)
```

> 6.244539773 seconds
> 106.290389523 seconds

### 3. sort (rdd.sortByKey().take(k))

```scala
cityPop.unpersist()
val tBegin = System.nanoTime
var res = cityPop.sortByKey(false).take(20)
println("exec in " + ((System.nanoTime - tBegin) / 1e9d))
res.foreach(println)
```

> 13.19124286 seconds
> 215.465674848 seconds

## 2. Spark is smart enough to just do a top k ?

```scala
// import org.apache.spark.sql.types._
// val schema = new StructType()
//   .add("Country", StringType, true)
//   .add("City", StringType, true)
//   .add("AccentCity", StringType, true)
//   .add("Region", StringType, true)
//   .add("Population", IntegerType, true)
//   .add("Latitude", StringType, true)
//   .add("Longitude", StringType, true)
// .schema(schema)
val dataDf = spark.read.format("csv").option("header", true).load("worldcitiespopHuge.txt")
val filterDf = dataDf.filter("Population is not null").withColumn("Population", col("Population").cast("Integer"))
filterDf.printSchema()
val cityPopDf = filterDf.select("City", "Population")
cityPopDf.show()
```

### good way (df.orderBy().limit(k))

```scala
cityPopDf.unpersist()
val tBegin = System.nanoTime
var res = cityPopDf.orderBy(desc("Population")).limit(20).show(20)
println("exec in " + ((System.nanoTime - tBegin) / 1e9d))
```

> 1.692267886 seconds
> 27.17035392 seconds
> 99.65037859 seconds

### force to order all data (df.orderBy().show(k))

```scala
cityPopDf.unpersist()
val tBegin = System.nanoTime
var res = cityPopDf.orderBy(desc("Population")).show(20)
println("exec in " + ((System.nanoTime - tBegin) / 1e9d))
```

> 1.665344476 seconds
> 25.29155218 seconds
> 100.705742925

### convert to rdd to do a top(;) (df.toRDD.top(k))

```scala
cityPopDf.unpersist()
val tBegin = System.nanoTime
var res = cityPopDf.rdd.map(x => (x(1).asInstanceOf[Int], x(0).asInstanceOf[String])).top(20)
println("exec in " + ((System.nanoTime - tBegin) / 1e9d))
res.foreach(println)
```

> 1.5592608 seconds
> 26.427353732 seconds
> 102.085210987
