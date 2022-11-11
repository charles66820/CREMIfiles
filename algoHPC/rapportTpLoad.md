# TP load balancing

## 0

On voit tous les paramètres (nombre de tâches, nombre de ressources, les tâches avec leur charge).
On voit le mapping trouvé et la répartition par ressource.
On voit la charge par ressource (la somme des charges de chaque tâches assignées).
On voit aussi les _metrics_ qui permettent d'évaluer les performances des algos.

## 1

Avec l'algo `Compact`, la répartition des tâches est plutôt moyenne pour les cas où il y a plutôt beaucoup de tâches et quand il y a peu de tâches et peu de ressources. On voit aussi que la répartition n'est pas bonne pour peu de tâches sur beaucoup de ressources.

| ressources $\downarrow$ |                  20  tâches                   |                   200 tâches                   |
| :---------------------: | :-------------------------------------------: | :--------------------------------------------: |
|            4            | ![Alt text](img/scenario_compact_T20_R4.png)  | ![Alt text](img/scenario_compact_T200_R4.png)  |
|           16            | ![Alt text](img/scenario_compact_T20_R16.png) | ![Alt text](img/scenario_compact_T200_R16.png) |

Avec l'algo `List_scheduler`, la répartition des tâches est vraiment bonne quand il y a beaucoup de tâches, moyenne avec peu de tâches et peu de ressources et assez inégale quand il y a peu de tâches et beaucoup de ressources.

| ressources $\downarrow$ |                      20  tâches                      |                      200 tâches                       |
| :---------------------: | :--------------------------------------------------: | :---------------------------------------------------: |
|            4            | ![Alt text](img/scenario_list_scheduler_T20_R4.png)  | ![Alt text](img/scenario_list_scheduler_T200_R4.png)  |
|           16            | ![Alt text](img/scenario_list_scheduler_T20_R16.png) | ![Alt text](img/scenario_list_scheduler_T200_R16.png) |

Avec l'algo `Round_robin`, la répartition des tâches n'est pas bonne avec beaucoup de ressources, moyenne avec beaucoup de tâches et peu de ressources et légèrement inégale quand il y a peu de tâches et peu de ressources.

| ressources $\downarrow$ |                    20  tâches                     |                     200 tâches                     |
| :---------------------: | :-----------------------------------------------: | :------------------------------------------------: |
|            4            | ![Alt text](img/scenario_round_robin_T20_R4.png)  | ![Alt text](img/scenario_round_robin_T200_R4.png)  |
|           16            | ![Alt text](img/scenario_round_robin_T20_R16.png) | ![Alt text](img/scenario_round_robin_T200_R16.png) |

Avec l'algo `Uniformly_random` la répartition des tâches n'est pas bonne avec beaucoup de ressources, pareil avec peu de tâches et peu de ressources et assez inégale quand il y a beaucoup de tâches et peu de ressources.

| ressources $\downarrow$ |                       20  tâches                       |                       200 tâches                        |
| :---------------------: | :----------------------------------------------------: | :-----------------------------------------------------: |
|            4            | ![Alt text](img/scenario_uniformly_random_T20_R4.png)  | ![Alt text](img/scenario_uniformly_random_T200_R4.png)  |
|           16            | ![Alt text](img/scenario_uniformly_random_T20_R16.png) | ![Alt text](img/scenario_uniformly_random_T200_R16.png) |

## 2

J'ai trouvé un cas compliqué (adversary case) pour le `list\_scheduler`. Ce cas est avec `8` tâches, `2` ressources. Les tâches ont les charges suivantes : `[4, 3, 3, 2, 4, 1, 3, 4]`. L'algo `list\_scheduler` répartit `14` charges sur la première ressource et `10` sur la seconde. Il existe une meilleure configuration avec `12` charges sur chaque ressources.

![Alt text](img/adversary_scenario_list_scheduler.png)

![Alt text](PXL_20221110_100114783.jpg)

## 3

J'ai complété la fonction `lpt` et elle passe le test unitaire.

Le `lpt` résout bien le cas compliqué que le `list\_scheduler` ne résout pas bien.

![Alt text](img/adversary_scenario_lpt.png)

## 4

Les algos `list_scheduler` et `lpt` sont assez similaires. On le constate quand on a beaucoup de tâches, par-contre l'algo `lpt` fonctionne mieux quand il y a peut de tâches et beaucoup de ressources.

|                   `lpt`                    |                   `list_scheduler`                    |
| :----------------------------------------: | :---------------------------------------------------: |
|  ![Alt text](img/scenario_lpt_T20_R4.png)  |  ![Alt text](img/scenario_list_scheduler_T20_R4.png)  |
| ![Alt text](img/scenario_lpt_T20_R16.png)  | ![Alt text](img/scenario_list_scheduler_T20_R16.png)  |
| ![Alt text](img/scenario_lpt_T200_R4.png)  | ![Alt text](img/scenario_list_scheduler_T200_R4.png)  |
| ![Alt text](img/scenario_lpt_T200_R16.png) | ![Alt text](img/scenario_list_scheduler_T200_R16.png) |

## 5

J'ai complété la fonction `lpt_with_limits` et elle passe le test unitaire.

## 6

Je n'ai pas trouvé de différence °~°.

|                   `lpt`                    |                    `list_scheduler`                    |
| :----------------------------------------: | :----------------------------------------------------: |
|  ![Alt text](img/scenario_lpt_T20_R4.png)  |  ![Alt text](img/scenario_lpt_with_limits_T20_R4.png)  |
| ![Alt text](img/scenario_lpt_T20_R16.png)  | ![Alt text](img/scenario_lpt_with_limits_T20_R16.png)  |
| ![Alt text](img/scenario_lpt_T200_R4.png)  | ![Alt text](img/scenario_lpt_with_limits_T200_R4.png)  |
| ![Alt text](img/scenario_lpt_T200_R16.png) | ![Alt text](img/scenario_lpt_with_limits_T200_R16.png) |

## 7

Je n'ai pas trouvé de cas compliqué (adversary case) °~°.

## 8

J'ai implémenté l'algo `list_scheduler_for_uniform_resources`, mon implémentation passe les tests grâce au saint `2`.
