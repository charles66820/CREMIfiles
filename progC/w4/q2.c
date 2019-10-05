int* create_array(unsigned int size) {
    int* array = (int*) malloc(size*sizeof(int));
    if(array==NULL) {
        fprintf(stderr, “Not enougt memory:\n”);
        exit(EXIT_FAILURE;
    }
    return array;
}

