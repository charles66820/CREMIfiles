# TP Batch scheduling

## 1

Done.

## 2

`ff` a un temps d'attente bien plus court que `fcfs` et vas environ `11` fois plus vite. Et le `makespan` et un peut mailleur.

```py
def ff(jobs, cluster, clock):
  for job in jobs:
    if cluster.available_nodes >= job.nodes:
      return (True, job)

  return (False, None)
```

## 3

`sjf` a un temps d'attente un peu plus court que `ff` et vas un peu plus vite. Et le `makespan` et un peut mailleur.

```py
def sjf(jobs, cluster, clock):
  selectedJob = None
  for job in jobs:
    if cluster.available_nodes >= job.nodes:
      if selectedJob == None or job.requested_run_time < selectedJob.requested_run_time:
        selectedJob = job

  return (selectedJob != None, selectedJob)
```

## 4

```py
def fcfs_easy(jobs, cluster, clock):
  nextjob = jobs[0]
  if cluster.available_nodes >= nextjob.nodes:
    return (True, nextjob)
  else:
    # select the first job finish that permit nextjob to be run
    curJobFirstEnd = None
    for curJob in cluster.running_jobs.values():
      if cluster.available_nodes + curJob.nodes >= nextjob.nodes:
        if curJobFirstEnd == None or curJob.expected_end < curJobFirstEnd.expected_end:
          curJobFirstEnd = curJob
    if curJobFirstEnd == None:
      return (False, None)

    # select a backfilling job to be run
    for job in jobs:
      # job can be execute when curJobFirstEnd finish
      if cluster.available_nodes >= job.nodes:
        # job can be execute before nextjob (not support execution in parallel with nextjob)
        if job.requested_run_time <= curJobFirstEnd.expected_end - clock:
          # nb nodes after nextjob start
          return (True, job)

  return (False, None)
```
