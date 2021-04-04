# Miniprojet compilation

## note

J'ai implémenté la boucle `for(Init ; Test ; Succ) Instr` avec les particularités suivante :

- Il est possible de définir une variable et de l'affecter dans la partie `init` de la boucle `for` (Ex `i : integer := 6` );
- Les environnements de variables locales sont empilés à chaque méthode, boucles et bloques.
  Les environnements son dépilés lorsque l'on sort de la méthode, la boucle ou le bloques.

Mon implémenté se trouve à niveau du commantaire `// NOTE: 1. for loop`
