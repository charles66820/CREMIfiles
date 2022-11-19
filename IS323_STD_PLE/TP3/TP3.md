# TP3

## notes

Start interactive shell :

```bash
spark-shell --master yarn
```

Submit an compile program :

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
val dataStingsList = dataClean.map(_.split(','))
```

## 2. count lines (observation)

Is `280280` observations.

```scala
dataStingsList.count()
```

## 3. Count the number of years

Is `118` years.

```scala
val years = dataStingsList.map(x => x(40)).distinct()
years.count()
```

## 4. The oldest and the most recent years of observation

The oldest years is `1662` and the most recent years is `1855`.

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

## 6.

Methods 1 :

```scala
dataStingsList.map(x => x(14))

distinct()
```

Methods 2 :

```
reduceByKey()
```

TODO: mesure time