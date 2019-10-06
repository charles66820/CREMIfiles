void deleteAllOccurencesOf(dynIntArray* t, int val) {
    if (t == NULL || (*t).tab == NULL || (*t).size > (*t).capacity) exit(EXIT_FAILURE);

    for (int i = (*t).size-1; i >= 0; i--)
        if ((*t).tab[i] == val)
            deleteValueAt(t, i);
}
