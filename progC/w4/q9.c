void deleteValueAt(dynIntArray* t, unsigned int ind) {
if (t == NULL || (*t).tab == NULL || (*t).size == NULL || (*t).size == 0 || (*t).size > (*t).capacity || ind > (*t).size) exit(EXIT_FAILURE);

    (*t).size--;

    // shift values
    for (int i = ind; i < (*t).size; i++) {
        (*t).tab[i] = (*t).tab[i+1];
    }
}
