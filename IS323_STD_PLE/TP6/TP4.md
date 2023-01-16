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

## Load data

```scala
val data = sc.textFile("/user/fzanonboito/CISD/worldcitiespop.txt")
val dataWithoutHeader = data.mapPartitionsWithIndex((i, r) => if (i == 0) r.drop(1) else r)
val dataStringsList = dataWithoutHeader.map(_.replace("\"", "").split(','))
val cityPop = dataStringsList.map(x => (try { x(4).toInt } catch {case e: Exception => 0}, x(1)))
// cityPop.top(40).foreach(println)
```

## 1. Best top k in spark

### 1. by hand (rdd.mapPartitions().top(k))

### 2. just top (rdd.top(k))

### 3. sort (rdd.sortByKey().take(k))

## 2. Spark is smart enough to just do a top k ?

### good way (df.orderBy().limit(k))

### force to order all data (df.orderBy().show(k))

### convert to rdd to do a top(;) (df.toRDD.top(k))
