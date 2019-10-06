void addValueAt(dynIntArray* t, int val, unsigned int ind) {
    if (t == NULL || (*t).tab == NULL || (*t).size > (*t).capacity || ind > (*t).size) exit(EXIT_FAILURE);

    // if table size is to small
    if ((*t).size == (*t).capacity) {
        // reallocation
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
        (*t).size = 0;
    }

    if ((*t).size > 0) {
        // shift values
        for (unsigned int i = (*t).size; i >= ind; i--) {
            (*t).tab[i+1] = (*t).tab[i];
        }
    }

    (*t).size++;
    // inset value in index
    (*t).tab[ind] = val;
}
