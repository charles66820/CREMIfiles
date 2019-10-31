add_executable(magasin magasin.c client.c)
target_link_libraries(magasin bcb sdd m)

install( TARGETS magasin RUNTIME DESTINATION bin )
