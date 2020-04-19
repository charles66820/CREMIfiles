#ifndef MODEL_H
#define MODEL_H

#include <SDL.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct Env_t Env;

#ifdef __ANDROID__
#define PRINTLOG(STR, ...) do { SDL_Log(STR, ##__VA_ARGS__);  } while(0)
#define ERRORLOG(STR, ...) do { SDL_LogError(SDL_LOG_CATEGORY_ERROR, STR, ##__VA_ARGS__); exit(EXIT_FAILURE); } while(0)
#else
#define PRINTLOG(STR, ...) do { printf(STR, ##__VA_ARGS__); } while(0)
#define ERRORLOG(STR, ...) do { fprintf(stderr, STR, ##__VA_ARGS__); exit(EXIT_FAILURE); } while(0)
#endif
#define PRINT(TITLE, STR, ...) do { char MSG[2000]; sprintf(MSG, STR, ##__VA_ARGS__); SDL_ShowSimpleMessageBox(0, TITLE, MSG, NULL); PRINTLOG(STR, ##__VA_ARGS__); } while (0)
#define ERROR(TITLE, STR, ...) do { char MSG[2000]; sprintf(MSG, STR, ##__VA_ARGS__); SDL_ShowSimpleMessageBox(0, TITLE, MSG, NULL); ERRORLOG(STR, ##__VA_ARGS__); } while (0)

#define APP_NAME "Recolor"
#define SCREEN_WIDTH 600
#define SCREEN_HEIGHT 600
#define DELAY 1000 / 60

Env* init(SDL_Window* win, SDL_Renderer* ren, int argc, char* argv[]);
void render(SDL_Window* win, SDL_Renderer* ren, Env* env);
void clean(SDL_Window* win, SDL_Renderer* ren, Env* env);
bool process(SDL_Window* win, SDL_Renderer* ren, Env* env, SDL_Event* e);

#endif
