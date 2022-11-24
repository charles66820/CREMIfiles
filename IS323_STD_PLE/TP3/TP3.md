# TP3

## notes

Start interactive shell :

```bash
spark-shell --master yarn
```

Submit a compiled program :

```bash
spark-submit --master yarn <program.jar|program.py> [--num-executors <nb>]
```

Dataset : `/user/fzanonboito/CISD/cliwoc15.csv`

"NA" (stands for "Not Available")

`map(x => (x,1))` = `map((_,1))`
`reduceByKey((x, y) => x + y)` = `reduceByKey(_+_)`

RegEx `\p{Punct}` matches any characters in the `Punct` script extension.
RegEx `\s` matchs any characters in `[\r\n\t\f\v ]`.

## 1. load & pre-process data

```scala
val data = sc.textFile("/user/fzanonboito/CISD/cliwoc15.csv")
val dataClean = data.mapPartitionsWithIndex((i, r) => if (i == 0) r.drop(1) else r)
val dataStingsList = dataClean.map(_.replace("\"", "").split(','))
```

## 2. count lines (observation)

It is `280280` observations.

```scala
dataStingsList.count()
```

## 3. Count the number of years

It is `118` years.

```scala
val years = dataStingsList.map(x => x(40)).distinct()
years.count()
```

## 4. The oldest and the most recent years of observation

The oldest year is `1662` and the most recent year is `1855`.

```scala
years.map(_.toInt).min()
years.map(_.toInt).max()
```

## 5. Years for min and max number of observations

The year with the minimum number of observations is `1747` with `4` observations.
The year with the maximum number of observations is `1778` with `8509` observations.

```scala
val yearsByNbObservations = dataStingsList.map(x => (x(40),1)).reduceByKey(_+_)
val nbObservationsByYears = yearsByNbObservations.map(x => (x._2, x._1))
nbObservationsByYears.min()
nbObservationsByYears.max()
```

## 6. Count the distinct departure places

> `14` is VoyageFrom column

Methods 1 `distinct` :

```scala
val tBegin = System.nanoTime
dataStingsList.map(x => x(14)).distinct().count()
val tEnd = (System.nanoTime - tBegin) / 1e9d

```

Methods 2 `reduceByKey` :

```scala
val tBegin = System.nanoTime
dataStingsList.map(x => (x(14), 1)).reduceByKey(_+_).count()
val tEnd = (System.nanoTime - tBegin) / 1e9d

```

I obtain `996` from the two methods.

The execution time of the two methods looks similar (`1.54` for Methods 1 and `1.60` for Methods 2).

## 7. the 10 most popular departure places

The result is :

| nb    | place      |
| :---- | :--------- |
| 25920 | Batavia    |
| 9757  | Rotterdam  |
| 8697  | Nederland  |
| 8680  | Montevideo |
| 8463  | La Coruña  |
| 8298  | Spithead   |
| 7906  | LA HABANA  |
| 7657  | LA CORUÑA  |
| 7522  | CÁDIZ      |
| 6713  | Nieuwediep |

```scala
val countedDeparturePlaces = dataStingsList.map(x => (x(14), 1)).reduceByKey(_+_)
countedDeparturePlaces.map(x => (x._2, x._1)).top(10).foreach(println)
```

## 8. the 10 roads the most often taken

First version

```scala
val countedRoads = dataStingsList.map(x => (x(14).toLowerCase.capitalize + "-" + x(15).toLowerCase.capitalize, 1)).reduceByKey(_+_)
countedRoads.map(x => (x._2, x._1)).top(10).foreach(println)
```

The first version result :

| nb   | roads                |
| :--- | :------------------- |
| 8514 | La coruña-Montevideo |
| 8459 | Montevideo-La coruña |
| 7525 | La coruña-La habana  |
| 7341 | Rotterdam-Batavia    |
| 6187 | Na-Na                |
| 6068 | La habana-La coruña  |
| 5256 | Nieuwediep-Batavia   |
| 5256 | Batavia-Rotterdam    |
| 4564 | Batavia-Nieuwediep   |
| 3996 | Nederland-Batavia    |

Second version

```scala
def roadsMap(x : Array[String]): (String, Int) = {
  val voyageFrom = x(14).toLowerCase.capitalize
  val voyageTo = x(15).toLowerCase.capitalize
  if (voyageFrom.compareTo(voyageTo) < 0)
    return (voyageFrom + "-" + voyageTo, 1)
  else
    return (voyageTo + "-" + voyageFrom, 1)
}
val countedRoads = dataStingsList.map(roadsMap).reduceByKey(_+_)
countedRoads.map(x => (x._2, x._1)).top(10).foreach(println)
```

The seconde version result :

| nb    | roads                  |
| :---- | :--------------------- |
| 16973 | La coruña-Montevideo   |
| 13593 | La coruña-La habana    |
| 12597 | Batavia-Rotterdam      |
| 9820  | Batavia-Nieuwediep     |
| 7530  | Batavia-Nederland      |
| 6187  | Na-Na                  |
| 3998  | Batavia-Texel          |
| 3164  | Amsterdam-Batavia      |
| 2661  | Batavia-Hellevoetsluis |
| 2231  | Cádiz-Montevideo       |

I have unified the roads name with `.toLowerCase.capitalize`.
I see my second version has been the sum of A-B and B-A roads (e.g. `La coruña-Montevideo` + `Montevideo-La coruña` = `8514` + `8459` = `16973`).
I see in my top the `NA` data.

## 9. the hottest month on average over the years

> `41` is Month column and `117` is ProbTair (temperatures) column

```scala
val dataDropNaTemp = dataStingsList.filter(x => x(117) != "NA")
// Count and sum all temperatures for each month
val dataTuple = dataDropNaTemp.map(x => (x(41), (x(117).toFloat, 1)))
val reducedData = dataTuple.reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))
// Compute the average over the years
val averageOverYears = reducedData.map(x => (x._2._1 / x._2._2, x._1))
averageOverYears.max()
//averageOverYears.top(12).foreach(println)
```

The hottest month is February with `22.98345`.

## 10. time for questions 1 to 4

### 1

We have a time of `~0.5` because we generate the DAG.

### 2

We have a time of `5.29` seconds because we load the data and we computed the data transformation for each line then we get the line number.
On the second, run we have a time of `2.60` seconds because the data transformation has been already done (and probably a cache effect).

### 3

We have a time of `3.84` seconds because we computed the years transformation (map and distinct) and we get the line number. We can note if we do not have any actions before this count we have a time of `6.60` because we have not loaded the data and we computed the data transformation.
On the second, run we have a time of `0.5` seconds because the data transformation has been already done.

### 4

We have the same time for `min` and `max`, which is `0.38` seconds.
And `6.13` seconds for first actions.

The reason is the same that the previous question.
