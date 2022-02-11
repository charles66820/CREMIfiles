for v in ompcol2 ompcol3 ompcol4 task1 task2 task3; do
  for g in 2 3 4 5; do
    echo -n "$v $g " 2>&1
    ./tsp-main 15 1234 $g $v > /dev/null
  done
done
