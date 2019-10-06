void addValueAsLast(dynIntArray* t, int val) {
    if (t == NULL || val == NULL || (*t).tab == NULL || (*t).size > (*t).capacity) exit(EXIT_FAILURE);

    if ((*t).size == (*t).capacity) {
        (*t).capacity *= 2;
        int *p_t = (*t).tab;
        (*t).tab = malloc((*t).capacity * sizeof(int));
        if ((*t).tab==NULL){
            fprintf(stderr,"Not enough memory!\n");
            exit(EXIT_FAILURE);
        }
        for (unsigned int i = 0; i<(*t).size; i++)
            (*t).tab[i] = p_t[i];
        free(p_t);
    }

    if ((*t).size == NULL) {
        (*t).size = 1;
    } else {
        (*t).size += 1;
    }
    (*t).tab[(*t).size-1] = val;
}
