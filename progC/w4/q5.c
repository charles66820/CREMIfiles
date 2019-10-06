void deleteLastValue(dynIntArray* t) {
    if (t == NULL || (*t).size == NULL || (*t).size < 0 || (*t).tab == NULL) exit(EXIT_FAILURE);
    (*t).size--;
}

