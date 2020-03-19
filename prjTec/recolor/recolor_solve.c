#include <libgen.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "game.h"
#include "game_io.h"
#include "solution.h"

typedef struct nb_color_s {
  uint* tab;
  uint tab_len;
} * nb_color;

nb_color nb_colors(game g) {
  if (g == NULL) {
    exit(EXIT_FAILURE);
  }

  uint* colors_tab = (uint*)malloc(16 * sizeof(uint));
  if (colors_tab == NULL) {
    game_delete(g);
    exit(EXIT_FAILURE);
  }
  uint* colors_ordre = (uint*)calloc(sizeof(uint), 16);
  if (colors_ordre == NULL) {
    game_delete(g);
    exit(EXIT_FAILURE);
  }

  uint cpt = 0;
  // I go through all the tab
  for (uint i = 0; i < game_height(g) * game_width(g); i++) {
    // check if the color is already in the tab, we don't add it
    bool exist = false;
    for (uint y = 0; y < cpt && y < 16; y++)  // (1)
      if (colors_tab[y] ==
          game_cell_current_color(g, i % game_width(g), i / game_width(g))){
        exist = true;
        colors_ordre[y]+=1;
        }

    // if the color isn't in the tab, we add it and we increment the cpt
    if (!exist) {
      colors_tab[cpt] =
          game_cell_current_color(g, i % game_width(g), i / game_width(g));
      cpt++;
      colors_ordre[cpt]=1;
    }  // We should do a realloc but It is not necessary in this exercise (1)
  }
  for(uint i=0;i<cpt;i++){
    for(uint j=i+1;j<cpt;j++){
        if ( colors_ordre[i] < colors_ordre[j] ) {
            uint c = colors_ordre[i];
            colors_ordre[i] = colors_ordre[j];
            colors_ordre[j] = c;
            c = colors_tab[i];
            colors_tab[i] = colors_tab[j];
            colors_tab[j] = c;
        }
    }
  }
  free(colors_ordre);
  // create struture for return colors and number nb_color *)of colors
  nb_color col_tab = malloc(sizeof(struct nb_color_s));
  if (col_tab == NULL) {
    game_delete(g);
    free(colors_tab);
    exit(EXIT_FAILURE);
  }
  uint* tab = (uint*)malloc((cpt + 1) * sizeof(uint));
  if (tab == NULL) {
    game_delete(g);
    free(colors_tab);
    free(col_tab);
    exit(EXIT_FAILURE);
  }

  for (uint i = 0; i < cpt + 1; i++) tab[i] = colors_tab[i];
  free(colors_tab);
  col_tab->tab = tab;
  col_tab->tab_len = cpt;
  return col_tab;
}

/**
 * @brief This fonction test all the possibles solutions for find the minimql
 *solution.
 * @param nb_colors structure with tab of colors (tab_colors) in game grid and
 * tab_colors length the colors number in game grid +1
 * @param size_sol the solution length (number of moves)
 * @param g the game will be test
 * @param solution an unsigne int table for store solution
 * @param k is k length of solution in recurcive call
 * @return if one solution as found or not
 * @pre @p nb_color is not NULL
 * @pre @p g is not NULL
 * @pre @p solution is not NULL
 **/
bool find_min_solution(nb_color nb_colors, uint size_sol, game g,
                       uint* solution, uint k) {
  // Recurcive call with k-1 length for make all posible solutions with all
  // colors
  for (uint i = 0; i < nb_colors->tab_len; i++) {
    solution[size_sol - k] = nb_colors->tab[i];  // add color to end of solution

    // check if solution work
    game gc = game_copy(g);
    game_play_one_move(gc, solution[size_sol - k]);
    if (game_is_over(gc)) {
      game_delete(gc);
      /* //Debug
      printf("comb :");
      for (uint i = 0; i < size_sol; i++) printf("%u", solution[i]);
      printf("\n"); //*/
      return true;
    }

    // On solution are completly create
    if (k != 0)
      if (find_min_solution(nb_colors, size_sol, gc, solution, k - 1)) {
        game_delete(gc);
        return true;
      }
    game_delete(gc);
  }
  return false;
}

/**
 * @brief This fonction test all the possibles solutions for found on solution.
 * @param nb_colors structure with tab of colors (tab_colors) in game grid and
 * tab_colors length the colors number in game grid +1
 * @param size_sol the solution length (number of moves)
 * @param g the game will be test
 * @param solution an unsigne int table for store solution
 * @param k is k length of solution in recurcive call
 * @return if one solution as found or not
 * @pre @p nb_color is not NULL
 * @pre @p g is not NULL
 * @pre @p solution is not NULL
 **/
bool find_one_solution(nb_color nb_colors, uint size_sol, game g,
                       uint* solution, uint k) {
  // Recurcive call with k-1 length for make all posible solutions with all
  // colors
  for (uint i = 0; i < nb_colors->tab_len; i++) {
    solution[size_sol - k] = nb_colors->tab[i];  // add color to end of solution

    // check if solution work
    game gc = game_copy(g);
    game_play_one_move(gc, solution[size_sol - k]);
    if (game_is_over(gc)) {
      game_delete(gc);
      /* //Debug
      printf("comb :");
      for (uint i = 0; i < size_sol; i++) printf("%u", solution[i]);
      printf("\n"); //*/
      return true;
    }

    // try next move
    if (k != 0)
      if (find_one_solution(nb_colors, size_sol, gc, solution, k - 1)){
        game_delete(gc);
        return true;
      }
    game_delete(gc);
  }
  return false;
}

uint count_valid_solution(nb_color nb_colors, uint size_sol, game g,
                          uint* solution, uint k) {
  // On solution are completly create °~°
  if (k == 0) return 0;  // TODO: 66
  // Recurcive call with k-1 length for make all posible solutions with all
  // colors
  uint nb = 0;
  for (uint i = 0; i < nb_colors->tab_len; i++) {
    solution[size_sol - k] = nb_colors->tab[i];  // add color to end of solution

    // check if solution work
    game gc = game_copy(g);
    game_play_one_move(gc, solution[size_sol - k]);
    if (game_is_over(gc)) {
      game_delete(gc);
      return 1;
    }

    // try next move
    nb += count_valid_solution(nb_colors, size_sol, gc, solution, k - 1);
    game_delete(gc);
  }
  return nb;
}

/**
 * @brief This fonction write solution in file with a new line.
 * @param filename file will be generate with the solution
 * @param responce the string with the solution or nb of solution
 * @pre @p filename is not NULL
 * @pre @p solution is not NULL
 **/
void save_sol_in_file(char* filename, char* responce) {
  if (filename == NULL || responce == NULL) {
    printf("At least one of the pointers is invalid\n");
    exit(EXIT_FAILURE);
  }
  uint filenamelen = (uint)strlen(filename) + 4;

  char* dir = malloc(sizeof(char) * filenamelen);
  if (dir == NULL) {
    printf("Not enough memory!\n");
    exit(EXIT_FAILURE);
  }
  strcpy(dir, filename);

  dirname(dir);
  if (strcmp(".", dir) && strcmp(filename, dir)) {
    char* mkcmd = malloc(sizeof(char) * filenamelen);
    if (mkcmd == NULL) {
      printf("Not enough memory!\n");
      free(dir);
      exit(EXIT_FAILURE);
    }
    sprintf(mkcmd, "mkdir -p %s", dir);
    system(mkcmd);
    free(mkcmd);
  }
  free(dir);

  FILE* savefile;
  savefile = fopen(filename, "w");
  if (savefile == NULL) {
    printf("The file couldn't be created\n");
    exit(EXIT_FAILURE);
  } else {
    fprintf(savefile, "%s\n", responce);
  }
  fclose(savefile);
}

/**
 * @brief find one possible solution and store it in the struct solution
 *
 * @param g game with cells to print
 * @return solution a struct with the solution or the msg "NO SOLUTION"
 */
solution find_one(game g) {
  if (g == NULL) {
    exit(EXIT_FAILURE);
  }
  solution the_solution = NULL;

  nb_color nb_col = nb_colors(g);
  uint nb_move = game_nb_moves_max(g);

  uint* sol = malloc(sizeof(uint) * nb_move);
  if (sol == NULL) {
    exit(EXIT_FAILURE);
  }

  /* #pragma region reverse color for fun
  uint* tmp = malloc(nb_col->tab_len * sizeof(uint));
  for (uint i = 0; i < nb_col->tab_len; i++) tmp[i] = nb_col->tab[i];
  for (uint i = 0; i < nb_col->tab_len; i++)
    nb_col->tab[i] = tmp[nb_col->tab_len - 1 - i];
  free(tmp);
  #pragma endregion */

  if (find_one_solution(nb_col, nb_move, g, sol, nb_move))
  the_solution = create_solution(sol, nb_move);

  free(nb_col->tab);
  free(nb_col);
  free(sol);
  game_delete(g);

  return the_solution;
}

/**
 * @brief seak for the number of solution
 *
 * @param g game with cells to print
 * @return uint the number of solutions for the game g
 */
uint nb_sol(game g) {
  if (g == NULL) {
    exit(EXIT_FAILURE);
  }
  uint nb_sol = 0;

  nb_color nb_col = nb_colors(g);
  uint nb_move = game_nb_moves_max(g);

  uint* sol = malloc(sizeof(uint) * nb_move);
  if (sol == NULL) {
    exit(EXIT_FAILURE);
  }
  // for (uint i = 0; i <= nb_move; i++)
  nb_sol += count_valid_solution(nb_col, nb_move, g, sol, nb_move);
  free(sol);

  game_delete(g);
  free(nb_col->tab);
  free(nb_col);

  return nb_sol;
}

/**
 * @brief find the solution who require the smallest amount of moves
 *
 * @param g game with cells to print
 * @return solution a struct with the smallest possible solution of the game g
 */
solution find_min(game g) {
  if (g == NULL) {
    exit(EXIT_FAILURE);
  }
  solution the_solution = NULL;

  nb_color nb_col = nb_colors(g);
  uint nb_move = game_nb_moves_max(g);

 /*  #pragma region reverse color for fun
  uint* tmp = malloc(nb_col->tab_len * sizeof(uint));
  for (uint i = 0; i < nb_col->tab_len; i++) tmp[i] = nb_col->tab[i];
  for (uint i = 0; i < nb_col->tab_len; i++)
    nb_col->tab[i] = tmp[nb_col->tab_len - 1 - i];
  free(tmp);
  #pragma endregion */

  uint* sol = malloc(sizeof(uint) * nb_move);
  if (sol == NULL) {
    exit(EXIT_FAILURE);
  }
  for (uint i = 0; i < nb_move; i++)
    if (find_min_solution(nb_col, i, g, sol, i)) {
      the_solution = create_solution(sol, i+1);
      break;
    }

  free(nb_col->tab);
  free(nb_col);
  free(sol);
  game_delete(g);

  return the_solution;
}
/* Appeler FIND_ONE avec nb_coups_max = 1; puis 2 puis 3 jusqu'à n*/

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "Error: invalid arguments");
    exit(EXIT_FAILURE);
  }

  solution retsol = NULL;
  game g = game_load(argv[2]);
  if (!strcmp(argv[1], "FIND_ONE"))
    retsol = find_one(g);
  else if (!strcmp(argv[1], "NB_SOL")) {
    char* buffer = malloc(sizeof(uint) * game_nb_moves_max(g));
    sprintf(buffer, "NB_SOL = %u", nb_sol(g));
    save_sol_in_file(strcat(argv[3], ".nbsol"), buffer);
    free(buffer);
    return EXIT_SUCCESS;
  } else if (!strcmp(argv[1], "FIND_MIN"))
    retsol = find_min(g);
  else {
    fprintf(stderr, "Error:  \"%s\" doesn't exist!\n", argv[1]);
    exit(EXIT_FAILURE);
  }

  // try if retsol is NULL else we can write in the file
  if (retsol != NULL) {
    char* s_sol = string_solution(retsol);
    save_sol_in_file(strcat(argv[3], ".sol"), s_sol);
    free(s_sol);
    delete_solution(retsol);
  } else
    save_sol_in_file(strcat(argv[3], ".sol"), "NO SOLUTION");

  return EXIT_SUCCESS;
}
