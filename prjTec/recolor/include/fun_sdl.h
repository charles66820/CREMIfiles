#ifndef FUN_SDL_H
#define FUN_SDL_H

#include <SDL.h>
#include <stdbool.h>
#include "game.h"

typedef struct color_cell {
  SDL_Rect rect;
  color color;
} COLOR_Cell;

typedef struct button {
  SDL_Rect rect;
  SDL_Texture* text;
  bool hover;
  bool pressed;
} BUTTON;

SDL_Color getColorFromGameColor(color c);
#if !defined(__ANDROID__)
SDL_Cursor* createPaintBucket(color c);
#endif
bool btnIsMouseHover(SDL_Event* e, SDL_Rect r);
bool btnIsFingerHover(SDL_Event* e, SDL_Rect r);
bool requestQuit(SDL_Window* win);

#endif
