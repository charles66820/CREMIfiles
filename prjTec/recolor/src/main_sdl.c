#include <SDL.h>
#include <SDL_image.h>
#include <SDL_ttf.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "SDL_model.h"

int main(int argc, char* argv[]) {
  // Initialize SDL2 and some extensions
  if (SDL_Init(SDL_INIT_VIDEO) != 0)
    ERRORLOG("Error: SDL_Init VIDEO (%s)\n", SDL_GetError());
  // IMG_INIT_PNG & IMG_INIT_PNG
  if (IMG_Init(IMG_INIT_PNG) != IMG_INIT_PNG)
    ERRORLOG("Error: IMG_Init PNG (%s)", SDL_GetError());
  if (TTF_Init() != 0) ERRORLOG("Error: TTF_Init (%s)\n", SDL_GetError());

  // create window and renderer
  SDL_Window* win = SDL_CreateWindow(
      APP_NAME, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH,
      SCREEN_HEIGHT, SDL_WINDOW_HIDDEN | SDL_WINDOW_RESIZABLE);
  if (!win) ERRORLOG("Error: SDL_CreateWindow (%s)\n", SDL_GetError());

  SDL_Renderer* ren = SDL_CreateRenderer(
      win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
  if (!ren) ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);
  if (!ren) ERROR("Error", "Error: SDL_CreateRenderer (%s)\n", SDL_GetError());

  SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);

  // Initialize your environment
  Env* env = init(win, ren, argc, argv);

  // Show window
  SDL_ShowWindow(win);

  // main render loop
  SDL_Event e;
  bool quit = false;
  while (!quit) {
    // Manage events
    while (SDL_PollEvent(&e)) {
      // process your events
      quit = process(win, ren, env, &e);
      if (quit) break;
    }

    // render all what you want
    render(win, ren, env);
    SDL_RenderPresent(ren);
    SDL_Delay(DELAY);
  }

  // clean your environment
  clean(win, ren, env);

  SDL_DestroyRenderer(ren);
  SDL_DestroyWindow(win);
  IMG_Quit();
  TTF_Quit();
  SDL_Quit();

  return EXIT_SUCCESS;
}
