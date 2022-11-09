# TP load balancing

## 0

On vois toutes les paramètres (nombre de tâches, nombre de resources, les tâches avec leurs charge).
On vois le mapping trouver et la repartition par resource.
On vois la charge par resource (la somme des charge de chaque tâches assigner).
On vois aussi les *metrics* qui permette d'évaluer les performances des algos.

## 1

Avec l'algo `Compact` la répartition des tâches est plutôt moyenne pour les cas où il y à plutôt beaucoup de tâches et quant il y à peu de tâches et peu de resources. On vois aussi que la répartition n'est pas bonne pour peu de tâches sur beaucoup de resources.

Algos `Compact` avec `20` tâches sur `4` resources :

![Alt text](study-load-balancing/scenario_compact_T20_R4.png)

Algos `Compact` avec `20` tâches sur `16` resources :

![Alt text](study-load-balancing/scenario_compact_T20_R16.png)

Algos `Compact` avec `200` tâches sur `4` resources :

![Alt text](study-load-balancing/scenario_compact_T200_R4.png)

Algos `Compact` avec `200` tâches sur `16` resources :

![Alt text](study-load-balancing/scenario_compact_T200_R16.png)

Avec l'algo `list_scheduler` la répartition des tâches est vraiment bonne quant il y à beaucoup de tâches, moyenne avec peu de tâches et peu de resources et moyenne bof quant il y a peu de tâches et beaucoup de resources.

Algos `list_scheduler` avec `20` tâches sur `4` resources :

![Alt text](study-load-balancing/scenario_list_scheduler_T20_R4.png)

Algos `list_scheduler` avec `20` tâches sur `16` resources :

![Alt text](study-load-balancing/scenario_list_scheduler_T20_R16.png)

Algos `list_scheduler` avec `200` tâches sur `4` resources :

![Alt text](study-load-balancing/scenario_list_scheduler_T200_R4.png)

Algos `list_scheduler` avec `200` tâches sur `16` resources :

![Alt text](study-load-balancing/scenario_list_scheduler_T200_R16.png)

Avec l'algo `round_robin` la répartition des tâches est n'est pas bonne avec beaucoup de resources, moyenne avec beaucoup de tâches et peu de resources et moyenne bof quant il y a peu de tâches et peu de resources.

Algos `round_robin` avec `20` tâches sur `4` resources :

![Alt text](study-load-balancing/scenario_round_robin_T20_R4.png)

Algos `round_robin` avec `20` tâches sur `16` resources :

![Alt text](study-load-balancing/scenario_round_robin_T20_R16.png)

Algos `round_robin` avec `200` tâches sur `4` resources :

![Alt text](study-load-balancing/scenario_round_robin_T200_R4.png)

Algos `round_robin` avec `200` tâches sur `16` resources :

![Alt text](study-load-balancing/scenario_round_robin_T200_R16.png)

Avec l'algo `uniformly_random` la répartition des tâches est n'est pas bonne avec beaucoup de resources, pareil avec peu de tâches et peu de resources et moyenne bof quand il y à beaucoup de tâches et peu de resources.

Algos `uniformly_random` avec `20` tâches sur `4` resources :

![Alt text](study-load-balancing/scenario_uniformly_random_T20_R4.png)

Algos `uniformly_random` avec `20` tâches sur `16` resources :

![Alt text](study-load-balancing/scenario_uniformly_random_T20_R16.png)

Algos `uniformly_random` avec `200` tâches sur `4` resources :

![Alt text](study-load-balancing/scenario_uniformly_random_T200_R4.png)

Algos `uniformly_random` avec `200` tâches sur `16` resources :

![Alt text](study-load-balancing/scenario_uniformly_random_T200_R16.png)

## 2


