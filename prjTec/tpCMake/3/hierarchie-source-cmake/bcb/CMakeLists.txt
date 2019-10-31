add_library(bcb chaine/chaine.c memoire/memoire.c)

install( TARGETS bcb ARCHIVE DESTINATION lib )
