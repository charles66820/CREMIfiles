#include "distributor.h"

static int indice = 0;

int distributor_next() { return indice++; }

int distributor_value() { return indice; }

void raz() { indice = 0; }
