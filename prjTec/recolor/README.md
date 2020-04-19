# Recolor

## Table of Contents

- [Recolor](#recolor)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Compilation](#compilation)
    - [Prerequisite](#prerequisite)
    - [Dependencies](#dependencies)
    - [Compile](#compile)
  - [Installation](#installation)
  - [Features](#features)
  - [Team](#team)

## Overview

This is the readme file about the game recolor.

Recolor is a game where you have to recolor the game until there is only one color left. you can change the color of the top-left square to propagate your color, everytime you change color, you add one move to the game, and all the color of the previous move change to the new color move.

It use SDL.

## Compilation

### Prerequisite

- Install gcc, make and cmake :
  - On Ubuntu :

    ```bash
    sudo apt update
    sudo apt install gcc make cmake
    ```

  - On windows follow [this tutorial](https://docs.google.com/document/d/1J9hmYZqJWYl5cPZbsa-0SUxm3aK9p-revsMnifJJuv4/edit?usp=sharing)

### Dependencies

- Install all dependencies on Ubuntu :

    ```bash
    sudo apt install libsdl2-dev libsdl2-image-dev  libsdl2-ttf-dev
    ```

### Compile

> Replace `Release` by `Debug` for debug mode

- On linux :

    ```bash
    mkdir build && cd build/
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make
    ```

- On Windows follow [this tutorial](https://docs.google.com/document/d/1J9hmYZqJWYl5cPZbsa-0SUxm3aK9p-revsMnifJJuv4/edit?usp=sharing) at the part "install SDL2, SDL2_IMG and SDL2_TTF".

    ```bsah
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles" -DCMAKE_C_COMPILER=C:/TDM-GCC-32/bin/gcc.exe ..
    make
    ```

> The generated binary / executable is in `generated` folder

## Installation

```bash
make install
```

## Features

- Recolor text version (recolor_text or recolor_text.exe)

  ```text
  This version of the game work on your terminal, it work with number, each number means one color.
  ```

- Recolor solve (recolor_solve or recolor_solve.exe)

  ```text
  this file solve one already existing game, you can search for 1 possible answer, the shortest answer, or the numbers of total possible answer.
  ```

- Recolor graphic version (recolor_sdl or recolor_sdl.exe)
  When the game is launch, choose a color to modify the game. Press r to restart or q to quit the game.

  ```text
  This is the graphic version of the game, it launch an executable who let you start a game, choose a color to make a move.
  ```

## Team

This version of the game was made by:

- Charles Gaudefroit

- Victor Andrault

- Arthur Blondeau
