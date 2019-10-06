/**
 * This function create an Array of int
 **/
dynIntArray* createArray(unsigned int capacity) {
    if (capacity == NULL || capacity == 0) exit(EXIT_FAILURE);

    dynIntArray *dia = malloc(sizeof(int)+sizeof(int)+sizeof(void*));
    if (dia == NULL) exit(EXIT_FAILURE);

    (*dia).capacity = capacity;
    (*dia).size = 0;
    (*dia).tab = (int *)malloc(capacity * sizeof(int));

    if ((*dia).tab == NULL) exit(EXIT_FAILURE);

    return dia;
}
