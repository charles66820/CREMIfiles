dynIntArray* copy(dynIntArray* t) {
    if (t == NULL || (*t).tab == NULL) exit(EXIT_FAILURE);

    dynIntArray * p = createArray((*t).capacity);
    if (p==NULL){
        fprintf(stderr,"Not enough memory!\n");
        return EXIT_FAILURE;
    }

    (*p).size = (*t).size;

    for (unsigned int i = 0; i < (*t).size; i++)
        (*p).tab[i] = (*t).tab[i];

    return p;
}
