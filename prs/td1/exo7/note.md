#

1. il ce pas que dans la sortie standard ou dans le fichier il y aura "hello \nworld" ce qui fait un saut de ligne.
2. dans stderr il n'y aura que "hello " car "world" est écrit sur 1 et non sur la constante STDOUT_FILENO.
3. `int main(){ printf("hello world"); return 1; }` ou `int main(){ write(STDOUT_FILENO, "hello world", 11); return 1; }`
