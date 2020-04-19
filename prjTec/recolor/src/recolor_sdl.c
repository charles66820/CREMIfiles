#include <SDL.h>
#include <SDL_image.h>
#include <SDL_ttf.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "SDL_model.h"
#include "fun_sdl.h"
#include "game.h"
#include "game_io.h"
#include "game_rand.h"
#include "load_game.h"

// Games assets and attributes
#define TRANSPARENCY 175 /* nuber between 0 and 255*/
#ifdef __ANDROID__
#define FONT_ROBOTO "Roboto-Regular.ttf"
#define FONT_OPENDYSLEXIC "OpenDyslexic-Regular.otf"
#define BACKGROUND "background.png"
#define BUTTON_SPRITE "button.png"
#define ICON "recolor.png"
#define SHADOWBOX1 "shadowBox1.png"
#define FONTSIZE 42
#define BTNWIDTH 375
#define BTNHEIGHT 100
#else
#define FONT_ROBOTO "assets/Roboto-Regular.ttf"
#define FONT_OPENDYSLEXIC "assets/OpenDyslexic-Regular.otf"
#define BACKGROUND "assets/background.png"
#define BUTTON_SPRITE "assets/button.png"
#define ICON "assets/recolor.png"
#define SHADOWBOX1 "assets/shadowBox1.png"
#define FONTSIZE 12
#define BTNWIDTH 150
#define BTNHEIGHT 40
#endif

struct Env_t {
  bool allowBackground;
  bool allowCursor;
  bool allowDyslexic;
  bool allowTransparency;
  SDL_Texture* background;
#if !defined(__ANDROID__)
  SDL_Cursor* cursor;
  SDL_Cursor* tempCursor;
#endif
  SDL_Surface* icon;
  SDL_Texture* button;
  TTF_Font* font;
  SDL_Texture* textWin;
  SDL_Texture* textLose;
  SDL_Texture* shadowBox1;
  game g;
  COLOR_Cell* cells;
  BUTTON btnRestart;
  BUTTON btnQuit;
};

Env* init(SDL_Window* win, SDL_Renderer* ren, int argc, char* argv[]) {
  Env* env = malloc(sizeof(struct Env_t));
  if (!env) ERROR("Game error", "Error: malloc (Not enought memory)\n");

  // Settings
  env->allowBackground = true;
#if !defined(__ANDROID__)
  env->allowCursor = true;
#endif
  env->allowDyslexic = false;
  env->allowTransparency = true;

  srand(time(0));

  // Init game
  env->g = NULL;
  env->g = load_game(argc, argv);

  // Load background texture
  if (env->allowBackground) {
    env->background = IMG_LoadTexture(ren, BACKGROUND);
    if (!env->background)
      ERROR("SDL error", "Error: IMG_LoadTexture (%s)\n", SDL_GetError());
  }

  // Load icon
  env->icon = IMG_Load(ICON);
  if (!env->icon) ERROR("SDL error", "Error: IMG_Load (%s)\n", SDL_GetError());

  // Load button texture
  env->button = IMG_LoadTexture(ren, BUTTON_SPRITE);
  if (!env->button)
    ERROR("SDL error", "Error: IMG_LoadTexture (%s)\n", SDL_GetError());

  // Load font
  env->font = TTF_OpenFont(env->allowDyslexic ? FONT_OPENDYSLEXIC : FONT_ROBOTO,
                           FONTSIZE);
  if (!env->font)
    ERROR("TTF error", "Error: TTF_OpenFont (%s)\n", SDL_GetError());

  // Load shadows
  env->shadowBox1 = IMG_LoadTexture(ren, SHADOWBOX1);
  if (!env->shadowBox1)
    ERROR("SDL error", "Error: IMG_Load (%s)\n", SDL_GetError());

  // Init text texture for win and lose
  TTF_SetFontStyle(env->font, TTF_STYLE_BOLD);
  SDL_Surface* s = TTF_RenderUTF8_Blended(
      env->font, "DOMMAGE", (SDL_Color){255, 0, 0, SDL_ALPHA_OPAQUE});
  env->textLose = SDL_CreateTextureFromSurface(ren, s);
  SDL_FreeSurface(s);

  s = TTF_RenderUTF8_Blended(env->font, "BRAVO",
                             (SDL_Color){0, 255, 0, SDL_ALPHA_OPAQUE});
  env->textWin = SDL_CreateTextureFromSurface(ren, s);
  SDL_FreeSurface(s);

  // Init button size and texts textures
  env->btnQuit.rect.w = BTNWIDTH;
  env->btnQuit.rect.h = BTNHEIGHT;
  env->btnRestart.rect.w = BTNWIDTH;
  env->btnRestart.rect.h = BTNHEIGHT;
  s = TTF_RenderUTF8_Blended(env->font, "Quit",
                             (SDL_Color){230, 92, 0, SDL_ALPHA_OPAQUE});
  env->btnQuit.text = SDL_CreateTextureFromSurface(ren, s);
  SDL_FreeSurface(s);

  s = TTF_RenderUTF8_Blended(env->font, "Restart",
                             (SDL_Color){153, 51, 255, SDL_ALPHA_OPAQUE});
  env->btnRestart.text = SDL_CreateTextureFromSurface(ren, s);
  SDL_FreeSurface(s);

  // Set icon
  SDL_SetWindowIcon(win, env->icon);

  // Init color grid cells
  env->cells =
      calloc(game_height(env->g) * game_width(env->g), sizeof(COLOR_Cell));

#if !defined(__ANDROID__)
  // Init cursors
  if (env->allowCursor) {
    env->tempCursor = createPaintBucket(0);
    if (!env->tempCursor)
      ERROR("SDL error", "Error: createPaintBucket (%s)\n", SDL_GetError());
    env->cursor = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_ARROW);
    SDL_SetCursor(env->cursor);
  }
#endif

  // Set buttons attribute to default value
  env->btnRestart.pressed = false;
  env->btnRestart.hover = false;
  env->btnQuit.pressed = false;
  env->btnQuit.hover = false;

  return env;
}

void render(SDL_Window* win, SDL_Renderer* ren, Env* env) {
  // set background color
  SDL_SetRenderDrawColor(ren, 250, 250, 250, SDL_ALPHA_OPAQUE);
  SDL_RenderClear(ren);

  // Local vars
  SDL_Texture* text;
  SDL_Surface* s;
  SDL_Rect rect;
  int winW = SCREEN_WIDTH;
  int winH = SCREEN_HEIGHT;
  int gameW = game_width(env->g);
  int gameH = game_height(env->g);

  SDL_Point mouse;
  SDL_GetMouseState(&mouse.x, &mouse.y);

  SDL_GetWindowSize(win, &winW, &winH);

  int xWinPadding =
      winW * 4 / 100;  // Window x padding is defind to 4% of window width
  int yWinPadding =
      winH * 4 / 100;  // Window y padding is defind to 4% of window height

  // Defined botton space
  int textheight = 0;
  SDL_QueryTexture(env->textWin, NULL, NULL, NULL,
                   &textheight);  // Get text height
  int buttonSpace = BTNHEIGHT + textheight;

  int gridMaxW =
      winW - xWinPadding * 2;  // Grid size is defind by window width -
                               // window padding left and right so * 2
  int gridMaxH =
      winH - (yWinPadding * 2 +
              buttonSpace);  // Grid size is defind by window height - window
                             // padding tob and bottom so * 2 + the botton space

  int cellSize =
      gridMaxW / gameW > gridMaxH / gameH ? gridMaxH / gameH : gridMaxW / gameW;

  int gridW = cellSize * gameW;
  int gridH = cellSize * gameH;

  int gridXPadding = gridW * 3 / 100;
  int gridYPadding = gridH * 3 / 100;

  cellSize = gridMaxW / gameW > gridMaxH / gameH
                 ? (gridMaxH - gridYPadding * 2) / gameH
                 : (gridMaxW - gridXPadding * 2) / gameW;

  int gridX = (winW - gridW) / 2;
  int gridY = (winH - (yWinPadding + buttonSpace) - gridH) / 2;

  // Draw background image
  if (env->allowBackground) {
    rect.x = 0;
    rect.y = 0;
    rect.w = winW;
    rect.h = winH - (yWinPadding + buttonSpace);
    SDL_RenderCopy(ren, env->background, NULL, &rect);
  }

  // Draw rounded surface for the color grid
  if (env->allowTransparency)
    SDL_SetTextureAlphaMod(env->shadowBox1, TRANSPARENCY);
  rect.x = gridX;
  rect.y = gridY;
  rect.w = gridW;
  rect.h = gridH;
  SDL_RenderCopy(ren, env->shadowBox1, NULL, &rect);

  // Create and draw grid cells
  for (int y = 0; y < gameH; y++)
    for (int x = 0; x < gameW; x++) {
      rect.x = cellSize * x + gridX + gridXPadding;
      rect.y = cellSize * y + gridY + gridYPadding;
      rect.w = cellSize;
      rect.h = cellSize;
      COLOR_Cell cell;
      cell.rect = rect;
      cell.color = game_cell_current_color(env->g, x, y);
      env->cells[(y * gameW) + x] = cell;

      SDL_Color c = getColorFromGameColor(cell.color);

      SDL_SetRenderDrawColor(
          ren, c.r, c.g, c.b,
          env->allowTransparency ? TRANSPARENCY : SDL_ALPHA_OPAQUE);
      SDL_RenderFillRect(ren, &rect);
    }

  // Draw line
  SDL_SetRenderDrawColor(ren, 0, 0, 0, SDL_ALPHA_OPAQUE);
  SDL_RenderDrawLine(ren, 0, winH - (yWinPadding + buttonSpace), winW,
                     winH - (yWinPadding + buttonSpace));

  // Draw game stats
  char* msg = malloc((50 + 78 * 2) * sizeof(char));
  if (!msg) ERROR("Game error", "Error: malloc (Not enought memory)\n");
  // 50 char in format + max char in uint * 2 uint
  sprintf(msg, "Nombre de coups joués / coups max: %u / %u",
          game_nb_moves_cur(env->g), game_nb_moves_max(env->g));

  s = TTF_RenderUTF8_Blended(env->font, msg,
                             (SDL_Color){61, 133, 198, SDL_ALPHA_OPAQUE});
  text = SDL_CreateTextureFromSurface(ren, s);
  SDL_FreeSurface(s);
  SDL_QueryTexture(text, NULL, NULL, &rect.w, &rect.h);
  rect.x = xWinPadding / 2;
  rect.y = winH - (yWinPadding + buttonSpace);
  SDL_RenderCopy(ren, text, NULL, &rect);
  SDL_DestroyTexture(text);
  free(msg);

  // Draw when game is lose or win
  if (game_nb_moves_cur(env->g) >= game_nb_moves_max(env->g) &&
      !game_is_over(env->g)) {
    rect.x = xWinPadding / 2;
    rect.y = winH - (yWinPadding + buttonSpace) + rect.h;
    SDL_QueryTexture(env->textLose, NULL, NULL, &rect.w, &rect.h);
    SDL_RenderCopy(ren, env->textLose, NULL, &rect);
  }
  if (game_is_over(env->g)) {
    rect.x = xWinPadding / 2;
    rect.y = winH - (yWinPadding + buttonSpace) + rect.h;
    SDL_QueryTexture(env->textWin, NULL, NULL, &rect.w, &rect.h);
    SDL_RenderCopy(ren, env->textWin, NULL, &rect);
  }

  // Draw buttons
  SDL_Rect rs = {0, 0, 0, 0};
  SDL_QueryTexture(env->button, NULL, NULL, &rs.w, &rs.h);
  rs.h = rs.h / 3;

  // Draw quit button
  env->btnQuit.rect.x = winW - xWinPadding / 2 - env->btnQuit.rect.w;
  env->btnQuit.rect.y = winH - (yWinPadding + buttonSpace) + rect.h;
  // Select btn sprite state
  rs.y = env->btnQuit.pressed ? rs.h * 2 : env->btnQuit.hover ? rs.h : 0;
  SDL_RenderCopy(ren, env->button, &rs, &env->btnQuit.rect);
  // Daw button text
  SDL_QueryTexture(env->btnQuit.text, NULL, NULL, &rect.w, &rect.h);
  rect.x = env->btnQuit.rect.x + ((env->btnQuit.rect.w - rect.w) / 2);
  rect.y = env->btnQuit.rect.y + ((env->btnQuit.rect.h - rect.h) / 2);
  SDL_RenderCopy(ren, env->btnQuit.text, NULL, &rect);

  // Draw restart button
  env->btnRestart.rect.x =
      winW - xWinPadding / 2 - env->btnRestart.rect.w - env->btnQuit.rect.w;
  env->btnRestart.rect.y = winH - (yWinPadding + buttonSpace) + rect.h;
  // Select btn sprite state
  rs.y = env->btnRestart.pressed ? rs.h * 2 : env->btnRestart.hover ? rs.h : 0;
  SDL_RenderCopy(ren, env->button, &rs, &env->btnRestart.rect);
  // Daw button text
  SDL_QueryTexture(env->btnRestart.text, NULL, NULL, &rect.w, &rect.h);
  rect.x = env->btnRestart.rect.x + ((env->btnRestart.rect.w - rect.w) / 2);
  rect.y = env->btnRestart.rect.y + ((env->btnRestart.rect.h - rect.h) / 2);
  SDL_RenderCopy(ren, env->btnRestart.text, NULL, &rect);
}

bool process(SDL_Window* win, SDL_Renderer* ren, Env* env, SDL_Event* e) {
  // Set buttons attributes to default value
  env->btnRestart.hover = false;
  env->btnQuit.hover = false;

  switch (e->type) {
    case SDL_QUIT:
      return requestQuit(win);
      break;
    case SDL_MOUSEMOTION:
    case SDL_FINGERMOTION:
#if !defined(__ANDROID__)
      // When color cell is hover change cursor to this color
      if (env->allowCursor) {
        bool onGrid = false;
        for (uint i = 0; i < game_height(env->g) * game_width(env->g); i++)
          if (btnIsMouseHover(e, env->cells[i].rect)) {
            SDL_SetCursor(env->tempCursor);
            SDL_FreeCursor(env->cursor);
            env->cursor = createPaintBucket(env->cells[i].color);
            if (!env->cursor)
              ERROR("SDL error", "Error: createPaintBucket (%s)\n",
                    SDL_GetError());
            SDL_SetCursor(env->cursor);
            onGrid = true;
            break;
          }
        if (onGrid) break;
      }
#endif

      // When button restart is hover the hover attribute is set to true
      if (btnIsMouseHover(e, env->btnRestart.rect) ||
          btnIsFingerHover(e, env->btnRestart.rect)) {
        env->btnRestart.hover = true;
#if !defined(__ANDROID__)
        goto cursor;
#endif
        break;
      }

      // When button quit is hover the hover attribute is set to true
      if (btnIsMouseHover(e, env->btnQuit.rect) ||
          btnIsFingerHover(e, env->btnQuit.rect)) {
        env->btnQuit.hover = true;
#if !defined(__ANDROID__)
      cursor:
        if (env->allowCursor) {
          SDL_FreeCursor(env->cursor);
          env->cursor = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_HAND);
          SDL_SetCursor(env->cursor);
        }
#endif
        break;
      }
#if !defined(__ANDROID__)
      if (env->allowCursor) {
        SDL_FreeCursor(env->cursor);
        env->cursor = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_ARROW);
        SDL_SetCursor(env->cursor);
      }
#endif
      break;
    case SDL_MOUSEBUTTONUP:
    case SDL_FINGERUP:
      // When color cell is clicked this color is play
      for (uint i = 0; i < game_height(env->g) * game_width(env->g); i++)
        if ((e->button.button == SDL_BUTTON_LEFT &&
             btnIsMouseHover(e, env->cells[i].rect)) ||
            btnIsFingerHover(e, env->cells[i].rect))
          if (env->cells[0].color != env->cells[i].color)
            game_play_one_move(env->g, env->cells[i].color);

      // When button restart is clicked the game is restart
      if ((e->button.button == SDL_BUTTON_LEFT &&
           btnIsMouseHover(e, env->btnRestart.rect)) ||
          btnIsFingerHover(e, env->btnRestart.rect))
        game_restart(env->g);

      // When button quit is clicked the game is quit
      if ((e->button.button == SDL_BUTTON_LEFT &&
           btnIsMouseHover(e, env->btnQuit.rect)) ||
          btnIsFingerHover(e, env->btnQuit.rect)) {
        env->btnQuit.pressed = false;
        return requestQuit(win);
      }

      // Set buttons pressed attribute to default value
      env->btnRestart.pressed = false;
      env->btnQuit.pressed = false;
      break;
    case SDL_MOUSEBUTTONDOWN:
    case SDL_FINGERDOWN:
      // When button restart is pressed the pressed attribute is set to true
      if ((e->button.button == SDL_BUTTON_LEFT &&
           btnIsMouseHover(e, env->btnRestart.rect)) ||
          btnIsFingerHover(e, env->btnRestart.rect))
        env->btnRestart.pressed = true;
      // When button quit is pressed the pressed attribute is set to true
      if ((e->button.button == SDL_BUTTON_LEFT &&
           btnIsMouseHover(e, env->btnQuit.rect)) ||
          btnIsFingerHover(e, env->btnQuit.rect))
        env->btnQuit.pressed = true;
      break;
    case SDL_KEYDOWN:
      switch (e->key.keysym.sym) {
        case SDLK_r:
          game_restart(env->g);
          break;
        case SDLK_q:
          return requestQuit(win);
          break;
        case SDLK_s:
          game_save(env->g, "data/quickSave.rec");
          break;
        default:
          break;
      }
      break;

    default:
      break;
  }

  return false;
}

void clean(SDL_Window* win, SDL_Renderer* ren, Env* env) {
  if (env->allowBackground) SDL_DestroyTexture(env->background);
#if !defined(__ANDROID__)
  if (env->allowCursor) SDL_FreeCursor(env->cursor);
  if (env->allowCursor) SDL_FreeCursor(env->tempCursor);
#endif
  SDL_FreeSurface(env->icon);
  SDL_DestroyTexture(env->button);
  SDL_DestroyTexture(env->btnQuit.text);
  SDL_DestroyTexture(env->btnRestart.text);
  TTF_CloseFont(env->font);
  SDL_DestroyTexture(env->textWin);
  SDL_DestroyTexture(env->textLose);
  SDL_DestroyTexture(env->shadowBox1);
  free(env->cells);
  game_delete(env->g);
  free(env);
}
