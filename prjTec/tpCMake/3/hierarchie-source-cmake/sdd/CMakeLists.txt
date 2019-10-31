add_library(sdd file/file.c paire/paire.c vext/vext.c)

install( TARGETS sdd ARCHIVE DESTINATION lib )

add_subdirectory(vext)
