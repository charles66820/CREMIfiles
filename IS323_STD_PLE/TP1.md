# TP1

## notes

jar folder : `/usr/hdp/current/hadoop-mapreduce-client/`

Compute pi :

```bash
yarn jar /usr/hdp/current/hadoop-mapreduce-client/hadoop-mapreduce-examples.jar pi 10000 1000
```

list tasks :

```bash
yarn application -list -appStates ALL
```

`/user/fzanonboito/CISD/worldcitiespop.txt`
`/user/fzanonboito/CISD/LesMiserables.txt`

## step 1 : First steps with HDFS

- done.
- done.
- `hdfs dfs -mkdir /user/$USER/data` done.
- `cat /dev/urandom > bigfile.txt` and `hdfs dfs -put bigfile.txt data`.
- my file is stored on 79 data nodes and in 7 chunks (blocks).
- done.

## step 2 : First steps with MapReduce

- done.
- done.

## step 3 : The Word Counter

- done.
- done.

```bash
yarn jar wordcounter-0.0.1.jar LesMiserables.txt worldCounterLM
```

- done. The result value of the reducer is store in `part-r-00000`. The jop exit statue is save as a file ex: `_SUCCESS`.
- The output data is 5 times less the input. It's mean the reducer work and aggregate all the words.
- done.
- We hae a smaller input with the combiner.

### Without combiner

```txt
FILE: Number of bytes read=4999680
FILE: Number of bytes written=10488391

Combine input records=0
Combine output records=0
Reduce input groups=51588
Reduce shuffle bytes=4999680
Reduce input records=421738
Reduce output records=51588
Spilled Records=843476
```

### With combiner

```txt
FILE: Number of bytes read=799813
FILE: Number of bytes written=2089033

Combine input records=421738
Combine output records=51588
Reduce input groups=51588
Reduce shuffle bytes=799813
Reduce input records=51588
Reduce output records=51588
Spilled Records=103176
```

## step 4 : MapReduce in Python

```bash
hadoop jar /usr/hdp/3.0.0.0-1634/hadoop-mapreduce/hadoop-streaming.jar \
-file mapreduce_wordcounter_python/mapper.py \
-mapper mapreduce_wordcounter_python/mapper.py \
-file mapreduce_wordcounter_python/reducer.py \
-reducer mapreduce_wordcounter_python/reducer.py \
-input LesMiserables.txt \
-output worldCounterLMJava
```

```bash
hadoop jar /usr/hdp/3.0.0.0-1634/hadoop-mapreduce/hadoop-streaming.jar \
-file mapreduce_wordcounter_python/mapper.py \
-mapper mapreduce_wordcounter_python/mapper.py \
-file mapreduce_wordcounter_python/reducer.py \
-reducer mapreduce_wordcounter_python/reducer.py \
-file mapreduce_wordcounter_python/reducer.py \
-combiner mapreduce_wordcounter_python/reducer.py \
-input LesMiserables.txt \
-output worldCounterLMJavaComB
```

- With python the execution time is more than 3 time bigger, use more memory and more CPU time.

## step 5 : World city populations

- done.
- 