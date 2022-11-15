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

En analysent les donnés j'ai vue qu'il n'y as pas de mots clef pour certains articles. J'ai décidé de drop les articles sans mots clef.

Mon programme prend trois arguments : le fichier avec les meta-données des articles, le dossier do sortie du top par décennies et le dossier de sortie du top globale.

Example de commande de lancement du program :

```bash
yarn jar topkeywords-0.0.1.jar /user/fzanonboito/CISD/IEEEdata.csv decadeTopOutput keywordTopOutput
```

Ma solution est divisé en 2 job.
Mon implémentation est composé de `2` jobs Map-Reduce, le premier `TopDecade` qui utilise `2` mappers et `1` reducer. Le second `TopKeyword` qui utilise `1` mappers et `1` reducer.

Le premier job récupère les mots clef et fait le top pour les décennies, pour cela j'ai le mapper et le reducer suivant :

- Mon mapper `RawDataMapper` traite les données en enlevents les lignes sans mots clef ou sans date et retourne en clef la décennies et en valeur un mots clef. Pour calculé la décennie je fais simplement une division sctricte par 10 pour récupéré la décennies par rapport a l'ans 0 (e.g. 1998 => 199, -212 => -21), cette solution est inexacte pour les date entre `-10` et `10` car cela donne la décennie `0` mais ce cas n'arrive pas dans nos données qui commance à partire de l'année `1962`. Il peut y avoir plusieurs mapper de ce type en parallel.

![Alt text](nullKw.png)

- Mon réduceur `DecadeReducer` compte pour chaque décennies le nombre de fois qu'un mots clef apparé puis fait le top de chaque mots clef. La sortie de ce réducer est sous la forme d'une ligne CSV séparé par des points virgule. Le premié élément est la décennie suivie du mots clef puis du nombre de papier ou il apparé dans la décennie et pour finir le top dans la décennie. Les lignes de chaque décennies sont trillé du plus féquant aux moins frécant puis par ordre alphabetique pour les mots clef par-contre les décennies peuvet être dans n'importe quelle ordre mais toute les linges d'une décennie ce suive. Il peut y avoir plusieurs reducer de ce type en parallel et chacun produit sont propre fichier. Je garde le nombre de papier où apparé le mot clef pour permetre le top global.

Le second job récupère les données par décennies du job précédent pour faire le top global. Pour faire ce top j'ai le mapper et le reducer suivant :

- Mon mapper `DecadeMapper` vérifie que les données sont bien présente et que nombre de papier est bien un entier ensuite il retourne en clef le mots clef et en valeur le nombre de papier par décennie ainsi que la décennie en question et le top dans la décennie. Il peut y avoir plusieurs mapper des ce type en parallel.

- Mon réduceur `KeywordReducer` fait la some du nombre de papier pour chaque mots clef, il regroupe aussi les données. Ensuiste il fait le top grace à la some total. Ce réducer retourne les donnée sous la forme d'une ligne CSV séparé par des points virgule. Cette ligne à pour éléments le mots clef, le top glibale, le nombre globale de papier ou apparé le mots ainsi que que chaque top et nomber papier par décennie. J'ai decidé de mettre le top de chaque décennie dans le fichier final et donc de gardé toute les lignes de tous les mots. Je pense que ce chois n'impacte pas les performance mais juste la taille du fichier final. Je touve qu'il est plus pratique de voire les top de chaque décennie dans ce fichier car il suffie de trillé la colonne voulue. Les donnés sont trillé du plus féquant aux moins frécant puis par ordre alphabetique pour les mots clef. Il peut y avoir qu'**un** seul reducer de ce type, c'est néssaisaire pour faire le top.

Example de où visualisé le fichier de sortie :

```bash
hdfs dfs -cat keywordTopOutput/part-r-00000
```

### step 2 : add new data

Pour prendre en compte l'ajout de nouveau papier publié après la première execution j'ai ajouté un quatième argument. Celui-ci permet de chargé les données des décennies déjà calculé.

Par example avec la commande si dessous on à les nouvelle donnés (`IEEE_Newdata.csv`) suivie des données déjà calculé (dans `decadeTopOutput`) puis le dossier qui recevra les décennies mis à jour et toujours le dossier de sortie final :

```bash
yarn jar topkeywords-0.0.1.jar IEEE_Newdata.csv decadeTopOutput decadeTopOutput_withNewData keywordTopOutput
```

Pour que cela fontionne il faut ajouté un mapper qui fait que chargé les donnés déjà calculé en plus des nouvlle donnés. Le reste devais fonctionné correctement.

TODO:
Pb copie sans modifications des décennie précédente dans le nouveaux dossier de sortie `decadeTopOutput_withNewData`.

TODO: do the modification.

## Report

### the expected size

J'ai calculé la taille des données aux quelle je m'attendé pour m'assuré du résultat.
Avec un script python j'ai déterminé que le nombre de mots clef différant est `130364`. J’obtiens `130365` avec le compteur (Reduce output records) de mon reducer `KeywordReducer`, il y à une différance de `1` car j'écrit une entéte dans le fichier de sortie.

J'ai aussi déterminé que le nombre de mots clef total (avec duplication) qui est `1060969`. J’obtiens `1060964` avec les compteurs (Map output records et Reduce input records) qui montre les données qui passe du mapper `DecadeMapper` au reducer `KeywordReducer`, il y à une différance de `5` mais je n'ai pas trouvé pourquoi.

### différant 

TODO: performance is expected to change if we increase/decrease

- analyze your solution: what is the expected size of intermediate and output data, how many Map-Reduce jobs were used and why, how many reducers were used in each job and why, how its performance is expected to change if we increase/decrease the number of available machines and/or papers, etc.

### perf int text

J'ai comparais les représentations intermédiaire des données. J'ai vue une seul différance qui est le nombre d'octets écrit, `21` octets de plus avec la version integer.

Conteur du nombre d'octets écrit par le job `TopDecade`. Version Text à gauche et version Integer à droit :

![textVsIntWritable](textVsIntWritable.png)

Par-contre je n'ai pas vue de différance significatif sur la performance, je ne sais pas si c'est du à mon implémentation on non. J'ai seulement fait 4 run (2 pour la version text et 2 pour la version integer).

TODO: if relevant, discuss the limitations of your solutions and how it could be improved.
