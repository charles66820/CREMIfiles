cmake_minimum_required(VERSION 2.6)

project(magasin C)

option(ENABLE_DEBUG ON)

set(CMAKE_INSTALL_PREFIX /net/cremi/cgoedefroit/espaces/travail/apps)

if(ENABLE_DEBUG)
    set(CMAKE_C_FLAGS "-g -Wall -std=c99")
else()
    set(CMAKE_C_FLAGS "-O3 -std=c99")
endif()

include_directories(bcb/chaine bcb/memoire magasin sdd/file sdd/paire sdd/vext)

#install(FILES bcb/chaine/chaine.h bcb/memoire/memoire.h magasin sdd/file/file.h sdd/paire/paire.h sdd/vex/vex.h DESTINATION include)

enable_testing()

add_subdirectory(bcb)
add_subdirectory(sdd)
add_subdirectory(magasin)
