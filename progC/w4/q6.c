void printArray(dynIntArray* t) {
    if (t==NULL || (*t).size == NULL || (*t).size < 0 || (*t).size > (*t).capacity || (*t).tab == NULL) exit(EXIT_FAILURE);

    for (int i = 0; i < (*t).size; i++)
        printf(" %d", (*t).tab[i]);
    printf("\n");
}
