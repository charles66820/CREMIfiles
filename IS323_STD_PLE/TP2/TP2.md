# TP2

## notes

jar folder : `/usr/hdp/current/hadoop-mapreduce-client/`

list tasks :

```bash
yarn application -list -appStates ALL
```

`/user/fzanonboito/CISD/IEEEdata.csv`

## Data processing

### step 1 : top 10 keywords

```bash
yarn jar topkeywords-0.0.1.jar /user/fzanonboito/CISD/IEEEdata.csv topkeywords_out
```



### step 2 : add new data

## Report

- present and explain your solution in details.
- analyze your solution: what is the expected size of intermediate and output data, how many Map-Reduce jobs were used and why, how many reducers were used in each job and why, how its performance is expected to change if we increase/decrease the number of available machines and/or papers, etc.
- discuss intermediate data representation: text vs. integer values (for counters). You may, for example, add some results that compare alternatives to support your choice.
- if relevant, discuss the limitations of your solutions and how it could be improved.
