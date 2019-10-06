/**
 * This function frees memory
**/
void freeArray(dynIntArray* t) {
    if (t != NULL) {
        int * tab = (*t).tab;
        free(tab);
        free(t);
    }
}
