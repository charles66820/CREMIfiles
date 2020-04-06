#!/bin/bash

cd jni

if [ ! -f SDL2-2.0.5.zip ]; then
    wget -c https://www.libsdl.org/release/SDL2-2.0.5.zip
    unzip SDL2-2.0.5.zip
    mv SDL2-2.0.5 SDL
fi

if [ ! -f SDL2_ttf-2.0.14.zip ]; then
    wget -c https://www.libsdl.org/projects/SDL_ttf/release/SDL2_ttf-2.0.14.zip
    unzip SDL2_ttf-2.0.14.zip
    mv SDL2_ttf-2.0.14 SDL_ttf
fi

if [ ! -f SDL2_image-2.0.1.zip ]; then
    wget -c https://www.libsdl.org/projects/SDL_image/release/SDL2_image-2.0.1.zip
    unzip SDL2_image-2.0.1.zip
    mv SDL2_image-2.0.1 SDL_image
fi

cd ..
