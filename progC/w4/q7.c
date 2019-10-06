unsigned int size(dynIntArray* t) {
    if (t == NULL || (*t).tab == NULL) exit(EXIT_FAILURE);
    return (*t).size;
}
