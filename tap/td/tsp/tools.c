#include "tools.h"

///////////////////////////////////////////////////////////////////////////
//
// Variables et fonctions internes (static) qui ne doivent pas être
// visibles à l'extérieur de ce fichier. À ne pas mettre dans le .h.
//
///////////////////////////////////////////////////////////////////////////

// nombres d'appels au dessin de la grille attendus par seconde
static unsigned long call_speed = 1 << 7;

static bool mouse_ldown = false;  // bouton souris gauche, vrai si enfoncé
static bool mouse_rdown = false;  // boutons souris droit, vrai si enfoncé
static bool oriented = false;     // pour afficher l'orientation de la tournée
static bool root = false;  // pour afficher le point de départ de la tournée
static int selectedVertex = -1;  // indice du point sélectionné avec la souris
static point *vertices = NULL;   // tableau de points
static int num_vertices = 0;     // nombre de points
static int mst = 3;              // pour drawGraph():
                                 // bit-0: dessin de l'arbre (1=oui/0=non)
                                 // bit-1: dessin de la tournée (1=oui/0=non)
static GLfloat sizePt = 5.0f;    // taille des points
static SDL_Window *window;
static SDL_GLContext glcontext;
static GLvoid *gridImage;
static GLuint texName;

static void drawLine(point p, point q) {
  glBegin(GL_LINES);
  glVertex2f(p.x, p.y);
  glVertex2f(q.x, q.y);
  glEnd();
}

static void drawEdge(point p, point q) {
  double linewidth = 1;
  glGetDoublev(GL_LINE_WIDTH, &linewidth);
  glBegin(GL_LINES);
  glVertex2f(p.x, p.y);
  glVertex2f(.2 * p.x + .8 * q.x, .2 * p.y + .8 * q.y);
  glEnd();
  glLineWidth(linewidth * 5);
  glBegin(GL_LINES);
  glVertex2f(.2 * p.x + .8 * q.x, .2 * p.y + .8 * q.y);
  glVertex2f(q.x, q.y);
  glEnd();
  glLineWidth(linewidth);
}

static void drawPoint(point p) {
  glPointSize(sizePt);
  glBegin(GL_POINTS);
  glVertex2f(p.x, p.y);
  glEnd();
}

// Convertit les coordonnées pixels en coordonnées dans le dessin
// https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluUnProject.xml
static void pixelToCoord(int pixel_x, int pixel_y, double *x, double *y) {
  GLint viewport[4];
  GLdouble proj[16];
  GLdouble modelview[16];

  // we query OpenGL for the necessary matrices etc.
  glGetIntegerv(GL_VIEWPORT, viewport);
  glGetDoublev(GL_PROJECTION_MATRIX, proj);
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview);

  GLdouble _X = pixel_x;
  GLdouble _Y = viewport[3] - pixel_y;
  GLdouble z;

  // using 1.0 for winZ give us a ray
  gluUnProject(_X, _Y, 1.0f, modelview, proj, viewport, x, y, &z);
}

// Récupère les coordonnées du centre de la fenêtre
static void getCenterCoord(double *x, double *y) {
  GLint viewport[4];
  glGetIntegerv(GL_VIEWPORT, viewport);
  pixelToCoord((viewport[0] + viewport[2]) / 2, (viewport[1] + viewport[3]) / 2,
               x, y);
}

static int getClosestVertex(double x, double y) {
  // renvoie l'indice i du point le plus proche de (x,y)
  int res = 0;
  double dmin = (x - vertices[0].x) * (x - vertices[0].x) +
                (y - vertices[0].y) * (y - vertices[0].y);

  for (int i = 1; i < num_vertices; i++) {
    double dist = (x - vertices[i].x) * (x - vertices[i].x) +
                  (y - vertices[i].y) * (y - vertices[i].y);
    if (dist < dmin) {
      dmin = dist;
      res = i;
    }
  }

  return res;
}

static char *getTitle(void) {
  static char buffer[100];
  sprintf(buffer,
          "Techniques Algorithmiques et Programmation - %d x %d (%.2lf)", width,
          height, scale);
  return buffer;
}

// Zoom d'un facteur s centré en (x,y)
static void zoomAt(double s, double x, double y) {
  glTranslatef(x, y, 0);
  glScalef(s, s, 1.0);
  glTranslatef(-x, -y, 0);
}

// Zoom d'un facteur s centré sur la position de la souris, modifie
// la variable globale scale du même facteur
static void zoomMouse(double s) {
  int mx, my;
  double x, y;
  SDL_GetMouseState(&mx, &my);
  pixelToCoord(mx, my, &x, &y);
  zoomAt(s, x, y);
  scale *= s;
}

// set drawGrid call speed
static void speedUp() {
  if (!call_speed) call_speed = 1;
  if ((call_speed << 1) != 0) call_speed <<= 1;
}

static void speedDown() {
  call_speed >>= 1;
  if (!call_speed) call_speed = 1;
}

void speedSet(unsigned long speed) {
  call_speed = speed;
  if (!call_speed) call_speed = 1;
}

static unsigned long speedMax() { return ULONG_MAX; }

bool NextPermutation(int *P, int n) {
  /*
    Génère la prochaine permutation P de taille n dans l'ordre
    lexicographique. On renvoie true si la prochaine permutation a pu
    être déterminée et false si P était la dernière permutation (et
    alors P n'est pas modifiée). Il n'est pas nécessaire que les
    valeurs de P soit dans [0,n[.

    On se base sur l'algorithme classique qui est:

    1. Trouver le plus grand index i tel que P[i] < P[i+1].
    S'il n'existe pas, la dernière permutation est atteinte.
    2. Trouver le plus grand indice j tel que P[i] < P[j].
    3. Echanger P[i] avec P[j].
    4. Renverser la suite de P[i+1] jusqu'au dernier élément.

  */
  int i = -1, j, m = n - 1, t;

  /* étape 1: cherche i le plus grand tq P[i]<P[i+1] */
  for (j = 0; j < m; j++)
    if (P[j] < P[j + 1]) i = j; /* on a trouvé un i tq P[i]<P[i+1] */
  if (i < 0) return false;      /* le plus grand i tq P[i]<[i+1] n'existe pas */

  /* étape 2: cherche j le plus grand tq P[i]<P[j] */
  for (j = i + 1; (j < n) && (P[i] < P[j]); j++)
    ;
  j--;

  /* étape 3: échange P[i] et P[j] */
  SWAP(P[i], P[j], t);

  /* étape 4: renverse P[i+1]...P[n-1] */
  for (++i; i < m; i++, m--) SWAP(P[i], P[m], t);

  return true;
}

// Vrai ssi (i,j) est sur le bord de la grille G.
static inline int onBorder(grid *G, int i, int j) {
  return (i == 0) || (j == 0) || (i == G->X - 1) || (j == G->Y - 1);
}

// Distance L2 entre s et t.
static inline double distL2(position s, position t) {
  return hypot(t.x - s.x, t.y - s.y);
}

typedef struct {
  // l'ordre de la déclaration est important
  GLubyte R;
  GLubyte G;
  GLubyte B;
} RGB;

static RGB color[] = {
    // l'ordre de la déclaration est important
    {0xE0, 0xE0, 0xE0},  // V_FREE
    {0x10, 0x10, 0x30},  // V_WALL
    {0xF0, 0xD8, 0xA8},  // V_SAND
    {0x00, 0x6D, 0xBA},  // V_WATER
    {0x7C, 0x70, 0x56},  // V_MUD
    {0x00, 0xA0, 0x60},  // V_GRASS
    {0x70, 0xE0, 0xD0},  // V_TUNNEL
    {0x80, 0x80, 0x80},  // M_NULL
    {0x12, 0x66, 0x66},  // M_USED
    {0x08, 0xF0, 0xF0},  // M_FRONT
    {0x90, 0x68, 0xF8},  // M_PATH
    {0xFF, 0x00, 0x00},  // C_START
    {0xFF, 0x88, 0x28},  // C_END
    {0x99, 0xAA, 0xCC},  // C_FINAL
    {0xFF, 0xFF, 0x80},  // C_END_WALL
    {0x66, 0x12, 0x66},  // M_USED2
    {0xC0, 0x4F, 0x16},  // C_FINAL2
    {0xFF, 0xFF, 0x00},  // C_PATH2
};

// nombre de couleurs dans color[]
static const int NCOLOR = (int)(sizeof(color) / sizeof(*color));

// Vrai ssi p est une position de la grille. Attention ! cela ne veut
// pas dire que p est un sommet du graphe, car la case peut contenir
// V_WALL.
static inline int inGrid(grid *G, position p) {
  return (0 <= p.x) && (p.x < G->X) && (0 <= p.y) && (p.y < G->Y);
}

//
// Construit l'image (pixels) de la grille à partir de la grille G. Le
// point (0,0) de G correspond au coin en haut à gauche.
//
// +--x
// |
// y
//
static void makeImage(grid *G) {
  // Attention! modifie update si fin=true
  static int cpt;  // compteur d'étape lorsqu'on reconstruit le chemin

  RGB *I = gridImage, c;
  int k = 0, v, m, f;

  // fin = true ssi le chemin a fini d'être construit, .start et .end
  //       ont été marqués M_PATH tous les deux
  bool const fin = (G->mark[G->start.x][G->start.y] == M_PATH) &&
                   (G->mark[G->end.x][G->end.y] == M_PATH);

  // debut = vrai ssi le chemin commence à être construit
  bool debut = false;
  for (int j = 0; j < G->Y && !debut; j++)
    for (int i = 0; i < G->X && !debut; i++)
      if (G->mark[i][j] == M_PATH) debut = true;

  if (fin) update = false;
  if (!debut) cpt = 0;
  if (debut) cpt++;
  if (debut && cpt == 1) speedSet(sqrt(call_speed / 4));

  double t1, t2, dmax = distL2(G->start, G->end);
  if (dmax == 0) dmax = 1E-10;  // pour éviter la division par 0

  for (int j = 0; j < G->Y; j++)
    for (int i = 0; i < G->X; i++) {
      m = G->mark[i][j];
      if ((m < 0) || (m >= NCOLOR)) m = M_NULL;
      v = G->value[i][j];
      if ((v < 0) || (v >= NCOLOR)) v = V_FREE;
      do {
        if (m == M_PATH) {
          c = color[m];
          if (fin && !erase) c = color[C_PATH2];
          break;
        }
        if (fin && erase) {
          c = color[v];
          break;
        }  // affiche la grille d'origine à la fin
        if (m == M_NULL) {
          c = color[v];
          break;
        }  // si pas de marquage
        if (m == M_USED || m == M_USED2) {
          // interpolation de couleur entre les couleurs M_USED(2) et
          // C_FINAL(2) ou bien M_USED(2) et v si on est en train de
          // reconstruire le chemin
          position p = {.x = i, .y = j};
          t1 = (m == M_USED) ? distL2(G->start, p) / dmax
                             : distL2(G->end, p) / dmax;
          t1 = fmax(t1, 0.0), t1 = fmin(t1, 1.0);
          t2 = (debut && erase) ? 0.5 * cpt / dmax : 0;
          t2 = fmin(t2, 1.0);
          f = (m == M_USED) ? C_FINAL : C_FINAL2;
          c.R = t2 * color[v].R +
                (1 - t2) * (t1 * color[f].R + (1 - t1) * color[m].R);
          c.G = t2 * color[v].G +
                (1 - t2) * (t1 * color[f].G + (1 - t1) * color[m].G);
          c.B = t2 * color[v].B +
                (1 - t2) * (t1 * color[f].B + (1 - t1) * color[m].B);
          break;
        }
        c = (m == M_NULL) ? color[v] : color[m];
        break;
      } while (0);
      I[k++] = c;
    }

  if (inGrid(G, G->start)) {
    k = G->start.y * G->X + G->start.x;
    I[k] = color[C_START];
  }

  if (inGrid(G, G->end)) {
    v = (G->value[G->end.x][G->end.y] == V_WALL) ? C_END_WALL : C_END;
    k = G->end.y * G->X + G->end.x;
    I[k] = color[v];
  }
}

//
// Alloue une grille aux dimensions x,y ainsi que son image. On force
// x,y>=3 pour avoir au moins un point qui n'est pas sur le bord.
//
static grid allocGrid(int x, int y) {
  grid G;
  position p = {-1, -1};
  G.start = G.end = p;
  if (x < 3) x = 3;
  if (y < 3) y = 3;
  G.X = x;
  G.Y = y;
  G.value = malloc(x * sizeof(*(G.value)));
  G.mark = malloc(x * sizeof(*(G.mark)));

  for (int i = 0; i < x; i++) {
    G.value[i] = malloc(y * sizeof(*(G.value[i])));
    G.mark[i] = malloc(y * sizeof(*(G.mark[i])));
    for (int j = 0; j < y; j++) G.mark[i][j] = M_NULL;  // initialise
  }

  gridImage = malloc(3 * x * y * sizeof(GLubyte));
  return G;
}

///////////////////////////////////////////////////////////////////////////
//
// Variables et fonctions utilisées depuis l'extérieur (non static).
// À mettre dans le .h.
//
///////////////////////////////////////////////////////////////////////////

// valeurs par défaut
int width = 640;
int height = 480;
bool update = true;
bool running = true;
bool hover = true;
bool erase = true;
double scale = 1;

char *TopChrono(const int i) {
#define CHRONOMAX 10
  /*
    Met à jour le chronomètre interne numéro i (i=0..CHRNONMAX-1) et
    renvoie sous forme de char* le temps écoulé depuis le dernier
    appel à la fonction pour le même chronomètre. La précision dépend
    du temps mesuré. Elle varie entre la minute et le 1/1000 de
    seconde. Plus précisément le format est le suivant:

    1d00h00'  si le temps est > 24h (précision: 1')
    1h00'00"  si le temps est > 60' (précision: 1s)
    1'00"0    si le temps est > 1'  (précision: 1/10s)
    1"00      si le temps est > 1"  (précision: 1/100s)
    0"000     si le temps est < 1"  (précision: 1/1000s)

    Pour initialiser et mettre à jour tous les chronomètres (dont le
    nombre vaut CHRONOMAX), il suffit d'appeler une fois la fonction,
    par exemple avec TopChrono(0). Si i<0, alors les pointeurs alloués
    par l'initialisation sont désalloués. La durée maximale est
    limitée à 100 jours. Si une erreur se produit (durée supérieure ou
    erreur avec gettimeofday()), alors on renvoie la chaîne
    "--error--".
  */
  if (i >= CHRONOMAX) return "--error--";

  /* variables globales, locale à la fonction */
  static int first =
      1; /* =1 ssi c'est la 1ère fois qu'on exécute la fonction */
  static char *str[CHRONOMAX];
  static struct timeval last[CHRONOMAX], tv;
  int j;

  if (i < 0) {  /* libère les pointeurs */
    if (!first) /* on a déjà alloué les chronomètres */
      for (j = 0; j < CHRONOMAX; j++) free(str[j]);
    first = 1;
    return NULL;
  }

  /* tv=temps courant */
  int err = gettimeofday(&tv, NULL);

  if (first) { /* première fois, on alloue puis on renvoie TopChrono(i) */
    first = 0;
    for (j = 0; j < CHRONOMAX; j++) {
      str[j] = malloc(
          10);  // assez grand pour "--error--", "99d99h99'" ou "23h59'59""
      last[j] = tv;
    }
  }

  /* t=temps en 1/1000" écoulé depuis le dernier appel à TopChrono(i) */
  long t = (tv.tv_sec - last[i].tv_sec) * 1000L +
           (tv.tv_usec - last[i].tv_usec) / 1000L;
  last[i] = tv;                        /* met à jour le chrono interne i */
  if ((t < 0L) || (err)) t = LONG_MAX; /* temps erroné */

  /* écrit le résultat dans str[i] */
  for (;;) { /* pour faire un break */
    /* ici t est en millième de seconde */
    if (t < 1000L) { /* t<1" */
      sprintf(str[i], "0\"%03li", t);
      break;
    }
    t /= 10L;        /* t en centième de seconde */
    if (t < 6000L) { /* t<60" */
      sprintf(str[i], "%li\"%02li", t / 100L, t % 100L);
      break;
    }
    t /= 10L;         /* t en dixième de seconde */
    if (t < 36000L) { /* t<1h */
      sprintf(str[i], "%li'%02li\"%li", t / 360L, (t / 10L) % 60L, t % 10L);
      break;
    }
    t /= 10L;         /* t en seconde */
    if (t < 86400L) { /* t<24h */
      sprintf(str[i], "%lih%02li'%02li\"", t / 3600L, (t / 60L) % 60L, t % 60L);
      break;
    }
    t /= 60L;         /* t en minute */
    if (t < 144000) { /* t<100 jours */
      sprintf(str[i], "%lid%02lih%02li'", t / 1440L, (t / 60L) % 24L, t % 60L);
      break;
    }
    /* error ... */
    sprintf(str[i], "--error--");
  }

  return str[i];
#undef CHRONOMAX
}

//
// Renvoie une position aléatoire de la grille qui est uniforme parmi
// toutes les valeurs de la grille du type t (hors les bords de la
// grille). Si aucune case de type t n'est pas trouvée, la position
// {-1,-1} est renvoyée.
//
position randomPosition(grid G, int t) {
  int i, j, c;
  int n;                       // n=nombre de cases de type t hors le bord
  int r = -1;                  // r=numéro aléatoire dans [0,n[
  position p = {-1, -1};       // position par défaut
  const int stop = G.X * G.Y;  // pour sortir des boucles
  const int x1 = G.X - 1;
  const int y1 = G.Y - 1;

  // On fait deux parcours: un 1er pour compter le nombre n de cases
  // de type t, et un 2e pour tirer au hasard la position parmi les
  // n. A la fin du premier parcours on connaît le nombre n de cases
  // de type t. On tire alors au hasard un numéro r dans [0,n[. Puis
  // on recommence le comptage (n=0) de cases de type t et on s'arrête
  // dès qu'on arrive à la case numéro r.

  c = 0;
  do {
    n = 0;
    for (i = 1; i < x1; i++)
      for (j = 1; j < y1; j++)
        if (G.value[i][j] == t) {
          if (n == r) {
            p = (position){i, j};
            i = j = stop;  // toujours faux au 1er parcours
          }
          n++;
        }
    c = 1 - c;
    if (c) r = random() % n;
  } while (c);  // vrai la 1ère fois, faux la 2e

  return p;
}

//
// Libère les pointeurs alloués par allocGrid().
//
void freeGrid(grid G) {
  for (int i = 0; i < G.X; i++) {
    free(G.value[i]);
    free(G.mark[i]);
  }
  free(G.value);
  free(G.mark);
  free(gridImage);
}

//
// Renvoie une grille de dimensions x,y rempli de points aléatoires de
// type et de densité donnés. Le départ et la destination sont
// initialisées aléatroirement dans une case V_FREE.
//
grid initGridPoints(int x, int y, int type, double density) {
  grid G = allocGrid(x, y);  // alloue la grille et son image

  // vérifie que le type est correct, M_NULL par défaut
  if ((type < 0) || (type >= NCOLOR)) type = M_NULL;

  // met les bords et remplit l'intérieur
  for (int i = 0; i < x; i++)
    for (int j = 0; j < y; j++)
      G.value[i][j] =
          onBorder(&G, i, j) ? V_WALL : ((RAND01 <= density) ? type : V_FREE);

  // position start/end aléatoires
  G.start = randomPosition(G, V_FREE);
  G.end = randomPosition(G, V_FREE);

  return G;
}

//
// Renvoie une grille aléatoire de dimensions x,y (au moins 3)
// correspondant à partir un labyrinthe qui est un arbre couvrant
// aléatoire uniforme. On fixe le point start = en bas à droit et end
// = en haut à gauche. La largeur des couloirs est donnée par w>0.
//
// Il s'agit de l'algorithme de Wilson par "marches aléatoires avec
// effacement de boucle" (cf. https://bl.ocks.org/mbostock/11357811)
//
grid initGridLaby(int x, int y, int w) {
  // vérifie les paramètres
  if (x < 3) x = 3;
  if (y < 3) y = 3;
  if (w <= 0) w = 1;

  // alloue la grille et son image
  int *value = malloc(x * y * sizeof(*value));

  // alloue la grille et son image
  grid Gw = allocGrid(x * (w + 1) + 1, y * (w + 1) + 1);

  // position par défaut
  Gw.start = (position){.x = Gw.X - 2, .y = Gw.Y - 2};
  Gw.end = (position){.x = 1, .y = 1};

  // au début des murs seulement sur les bords
  for (int i = 0; i < Gw.X; i++) {
    for (int j = 0; j < Gw.Y; j++) {
      Gw.value[i][j] =
          ((i % (w + 1) == 0) || (j % (w + 1) == 0)) ? V_WALL : V_FREE;
    }
  }

  for (int i = 0; i < x; i++)
    for (int j = 0; j < y; j++) value[i * y + j] = -1;

  int count = 1;
  value[0] = 0;
  while (count < x * y) {
    int i0 = 0;
    while (i0 < x * y && value[i0] != -1) i0++;
    value[i0] = i0 + 1;
    while (i0 < x * y) {
      int x0 = i0 / y;
      int y0 = i0 % y;
      while (true) {
        int dir = random() & 3;  // pareil que random()%4
        switch (dir) {
          case 0:
            if (x0 <= 0) continue;
            x0--;
            break;
          case 1:
            if (y0 <= 0) continue;
            y0--;
            break;
          case 2:
            if (x0 >= x - 1) continue;
            x0++;
            break;
          case 3:
            if (y0 >= y - 1) continue;
            y0++;
            break;
        }
        break;
      }
      if (value[x0 * y + y0] == -1) {
        value[x0 * y + y0] = i0 + 1;
        i0 = x0 * y + y0;
      } else {
        if (value[x0 * y + y0] > 0) {
          while (i0 != x0 * y + y0 && i0 > 0) {
            int i1 = value[i0] - 1;
            value[i0] = -1;
            i0 = i1;
          }
        } else {
          int i1 = i0;
          i0 = x0 * y + y0;
          do {
            int x0 = i0 / y;
            int y0 = i0 % y;
            int x1 = i1 / y;
            int y1 = i1 % y;
            if (x0 < x1)
              for (int i = 0; i < w; ++i)
                Gw.value[x1 * (w + 1)][y0 * (w + 1) + i + 1] = V_FREE;
            if (x0 > x1)
              for (int i = 0; i < w; ++i)
                Gw.value[x0 * (w + 1)][y0 * (w + 1) + i + 1] = V_FREE;
            if (y0 < y1)
              for (int i = 0; i < w; ++i)
                Gw.value[x1 * (w + 1) + i + 1][y1 * (w + 1)] = V_FREE;
            if (y0 > y1)
              for (int i = 0; i < w; ++i)
                Gw.value[x1 * (w + 1) + i + 1][y0 * (w + 1)] = V_FREE;
            i0 = i1;
            i1 = value[i0] - 1;
            value[i0] = 0;
            count++;
          } while (value[i1] != 0);
          break;
        }
      }
    }
  }

  free(value);

  return Gw;
}

grid initGridFile(char *file) {
  FILE *f = fopen(file, "r");
  if (f == NULL) {
    printf("Cannot open file \"%s\"\n", file);
    exit(1);
  }

  char *L = NULL;  // L=buffer pour la ligne de texte à lire
  size_t b = 0;    // b=taille du buffer L utilisé (nulle au départ)
  ssize_t n;       // n=nombre de caractères lus dans L, sans le '\0'

  // Etape 1: on évalue la taille de la grille. On s'arrête si c'est
  // la fin du fichier ou si le 1ère caractère n'est pas un '#'

  int x = 0;  // x=nombre de caractères sur une ligne
  int y = 0;  // y=nombre de lignes

  while ((n = getline(&L, &b, f)) > 0) {
    if (L[0] != '#') break;
    if (L[n - 1] == '\n') n--;  // se termine par '\n' sauf si fin de fichier
    if (n > x) x = n;
    y++;
  }

  rewind(f);

  if (x < 3) x = 3;
  if (y < 3) y = 3;
  grid G = allocGrid(x, y);

  // met des bords et remplit l'intérieur
  for (int i = 0; i < x; i++)
    for (int j = 0; j < y; j++)
      G.value[i][j] = onBorder(&G, i, j) ? V_WALL : V_FREE;

  // Etape 2: on relie le fichier et on remplit la grille

  int v;
  for (int j = 0; j < y; j++) {
    n = getline(&L, &b, f);
    if (L[n - 1] == '\n') n--;     // enlève le '\n' éventuelle
    for (int i = 0; i < n; i++) {  // ici n<=x
      switch (L[i]) {
        case ' ':
          v = V_FREE;
          break;
        case '#':
          v = V_WALL;
          break;
        case ';':
          v = V_SAND;
          break;
        case '~':
          v = V_WATER;
          break;
        case ',':
          v = V_MUD;
          break;
        case '.':
          v = V_GRASS;
          break;
        case '+':
          v = V_TUNNEL;
          break;
        case 's':
          v = V_FREE;
          G.start = (position){.x = i, .y = j};
          break;
        case 't':
          v = V_FREE;
          G.end = (position){.x = i, .y = j};
          break;
        default:
          v = V_FREE;
      }
      G.value[i][j] = v;
    }
  }

  free(L);
  fclose(f);
  return G;
}

void addRandomBlob(grid G, int type, int n) {
  // ne touche pas au bord de la grille: 0, G.X-1 et G.Y-1
  // ni à .start et .end
  int V[8][2] = {{0, -1},  {1, 0},  {0, 1},  {-1, 0},
                 {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

  for (int i = 0; i < n; i++)  // met n graines
    G.value[1 + random() % (G.X - 2)][1 + random() % (G.Y - 2)] = type;

  int m = (G.X + G.Y) / 2;
  for (int t = 0; t < m; t++)  // répète m fois
    for (int i = 1; i < G.X - 1; i++)
      for (int j = 1; j < G.Y - 1; j++) {
        int c = 0;  // c = nombre de voisins à "type"
        for (int k = 0; k < 8; k++)
          if (G.value[i + V[k][0]][j + V[k][1]] == type) c++;
        if (c && random() % ((8 - c) * 20 + 1) == 0) G.value[i][j] = type;
      }
}

static inline double angle(double const x, double const y)
/*
  Renvoie l'angle de [0,2𝜋[ en radian du point de coordonnées
  cartésiennes (x,y) par rapport à l'axe des abscisses (1,0). NB:
  atan(y/x) donne un angle [-𝜋/2,+𝜋/2] ce qui n'est pas ce que l'on
  veut. On renvoie 0 si (x,y)=(0,0).
*/
{
#define M_2PI 6.28318530717958647692528676655900576839 /* constante 2𝜋 */

  if (x == 0) {
    if (y > 0) return M_PI_2;
    if (y < 0) return M_PI_2 + M_PI;
    return 0;
  }

  // atan(y/x) renvoie un angle entre -𝜋/2 et +𝜋/2
  // l'angle est correct si x>0 et y>0
  // si x,y de signe contraire alors atan(y/x)=-atan(y/x)

  double const a = atan(y / x);

  if (x > 0) {
    if (y > 0) return a;
    return a + M_2PI;
  }

  return a + M_PI;
}

void addRandomArc(grid G, int type, int n) {
  // pour ajouter n segments (arcs) de textures t entre .star et .end
  // ne touche pas au bord de la grille: 0, G.X-1 et G.Y-1
  // ni à .start et .end sauf s'ils sont <0, dans ce
  // cas les met à random dans V_FREE
  //
  // Algo: n fois on crée un arc de cercle entre +𝜋/4 et -𝜋/4 (en fait
  // 𝜋/(2c) où c=0.5 environ) autour de l'axe s-t (on tire au hasard si
  // l'arc est depuis s ou depuis t). L'arc est continue selon de
  // 4-voisinage.

  if (!inGrid(&G, G.start) || !inGrid(&G, G.end)) {
    G.start = randomPosition(G, V_FREE);
    G.end = randomPosition(G, V_FREE);
  }
  // ici .star et .end sont dans la grille
  if (G.start.x == G.end.x && G.start.y == G.end.y) return;  // rien à faire

  double d, a1, a2, t, a, da;
  position p, q, u, v;
  double const c = 0.6;

  for (int i = 0; i < n; i++) {  // on répète n fois

    a1 = RAND01 * M_PI * c - M_PI * c / 2;  // angle1 par rapport au segment s-t
    a2 = RAND01 * M_PI * c - M_PI * c / 2;  // angle2 par rapport au segment s-t

    if (a1 < a2) SWAP(a1, a2, t);                // ici a1>a2
    if (a1 - a2 > M_PI / 3) a2 = (a1 + a2) / 2;  // si arc trop grand

    // choisit au hasard le point de départ: s ou t
    p = G.start, q = G.end;
    if (random() & 1) SWAP(p, q, u);

    d = RAND01 * distL2(p, q);        // d=distance p-q
    t = angle(q.x - p.x, q.y - p.y);  // angle entre p->q
    a = a1;                           // angle courant
    da = (a1 - a2) / (1.5 * d);       // variation d'angle

    for (int j = 0; j < (int)(1.5 * d); j++) {
      // u=position courante à dessiner
      u = (position){p.x + d * cos(t + a), p.y + d * sin(t + a)};
      if (!inGrid(&G, u) || onBorder(&G, u.x, u.y)) continue;
      G.value[u.x][u.y] = type;  // position sur la grille
      a -= da;                   // pour la prochaine position

      // teste si la prochaine position v de u va être en diagonale
      // pour la 4-connexité
      v = (position){p.x + d * cos(t + a), p.y + d * sin(t + a)};
      if (abs(u.x - v.x) > 0 && abs(u.y - v.y) > 0) {  // v en diagonale ?
        if (random() & 1)
          u.x = v.x;
        else
          u.y = v.y;  // corrige x ou y au hasard
        if (!inGrid(&G, u) || onBorder(&G, u.x, u.y)) continue;
        G.value[u.x][u.y] = type;  // position sur la grille
      }
    }
  }

  return;
}

// Initialisation de SDL
void init_SDL_OpenGL(void) {
  SDL_Init(SDL_INIT_VIDEO);
  window = SDL_CreateWindow(
      getTitle(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width,
      height, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

  if (window == NULL) {  // échec lors de la création de la fenêtre
    printf("Could not create window: %s\n", SDL_GetError());
    SDL_Quit();
    exit(1);
  }

  // SDL_CreateRenderer(window,-1,SDL_RENDERER_SOFTWARE);
  SDL_GetWindowSize(window, &width, &height);
  // Contexte OpenGL
  glcontext = SDL_GL_CreateContext(window);

  // Projection de base, un point OpenGL == un pixel
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, width, height, 0.0, 0.0f, 1.0f);
  glScalef(scale, scale, 1.0);

  // Some GL options
  glEnable(GL_LINE_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glGenTextures(1, &texName);
  glBindTexture(GL_TEXTURE_2D, texName);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}

// Fermeture de SDL
void cleaning_SDL_OpenGL() {
  SDL_GL_DeleteContext(glcontext);
  SDL_DestroyWindow(window);
  SDL_Quit();
}

// Génère n points aléatoires du rectangle [0,width] × [0,height] et
// renvoie le tableau des n points (type double) ainsi générés. Met à
// jour les variables globales vertices[] et num_vertices.
point *generatePoints(int n) {
  vertices = malloc(n * sizeof(point));
  const double rx = (double)width / RAND_MAX;
  const double ry = (double)height / RAND_MAX;
  for (int i = 0; i < n; i++) {
    vertices[i].x = random() * rx;
    vertices[i].y = random() * ry;
  }
  num_vertices = n;
  return vertices;
}

// Génère n points du rectangle [0,width] × [0,height] répartis
// aléatoirement sur k cercles concentriques (de rayon height/2.2)
point *generateCircles(int n, int k) {
  point c = {width / 2.0, height / 2.0};  // centre
  vertices = malloc(n * sizeof(point));
  const double r0 = ((width < height) ? width : height) / (2.2 * k);

  for (int i = 0; i < n; i++) {  // on place les n points
    int j = random() % k;        // j=numéro du cercle
    double r = (j + 1) * r0;
    const double K = 2.0 * M_PI / RAND_MAX;
    double a = K * random();
    vertices[i].x = c.x + r * cos(a);
    vertices[i].y = c.y + r * sin(a);
  }

  num_vertices = n;
  return vertices;
}

// couleurs pour drawX()

#define FOND_NOIR 0.0, 0.0, 0.0   // couleur du fond
#define ORANGE 0.9, 0.8, 0.3      // racine, point de départ
#define VERT_FONCE 0.0, 0.4, 0.0  // arête de l'arbre
#define ROUGE 1.0, 0.0, 0.0       // point
#define BLANC 1.0, 1.0, 1.0       // ligne de la tournée
#define VERT 0.0, 1.0, 0.0        // chemin

// dessine les k premiers sommets d'une tournée; ou
// dessine une tournée complète (si k=n+1); ou
// dessine un graphe (G<>NULL) et sa tournée complète.

void drawX(point *V, int n, int *P, int k, graph *G) {
  static unsigned int last_tick = 0;

  // saute le dessin si le précédent a été fait il y a moins de 20ms
  // ou si update est faux
  if ((!update) && (last_tick + 20 > SDL_GetTicks())) return;
  last_tick = SDL_GetTicks();

  // gestion de la file d'event
  handleEvent(false);

  // efface la fenêtre
  glClearColor(FOND_NOIR, 1);
  glClear(GL_COLOR_BUFFER_BIT);

  // dessine G, s'il existe
  if (G && V && G->list && (G->deg[0] >= 0) && (mst & 1)) {
    glLineWidth(5.0);
    glColor3f(VERT_FONCE);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < G->deg[i]; j++)
        if (i < G->list[i][j]) drawLine(V[i], V[G->list[i][j]]);
    glLineWidth(1.0);
  }

  // dessine le cycle en blanc si k>n; ou
  // dessine le chemin en vert
  if (V && P && (P[0] >= 0) && ((G && (mst & 2)) || (!G && (mst & 1)))) {
    if (k > n)
      glColor3f(BLANC);
    else
      glColor3f(VERT);
    for (int i = 0; i < k - 1; i++) {
      if (oriented)
        drawEdge(V[P[i]], V[P[(i + 1) % n]]);
      else
        drawLine(V[P[i]], V[P[(i + 1) % n]]);
    }
    if (root) {
      glColor3f(ORANGE);
      if (oriented && (n > 0))
        drawEdge(V[P[0]], V[P[1]]);
      else
        drawLine(V[P[0]], V[P[1]]);
    }
  }

  // dessine les points
  if (V) {
    glEnable(GL_POINT_SMOOTH);  // pour avoir des points ronds
    glColor3f(ROUGE);
    for (int i = 0; i < n; i++) drawPoint(V[i]);
    if (root && P && (P[0] >= 0)) {
      glColor3f(ORANGE);
      drawPoint(V[P[0]]);
    }
    glDisable(GL_POINT_SMOOTH);  // pour avoir des points carrés
  }

  // affiche le dessin
  SDL_GL_SwapWindow(window);
}

void drawTour(point *V, int n, int *P) { drawX(V, n, P, n + 1, NULL); }
void drawPath(point *V, int n, int *P, int k) { drawX(V, n, P, k, NULL); }
void drawGraph(point *V, int n, int *P, graph G) { drawX(V, n, P, n + 1, &G); }

static void drawGridImage(grid G) {
  // Efface la fenêtre
  glClearColor(FOND_NOIR, 1);
  glClear(GL_COLOR_BUFFER_BIT);

  // Dessin
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, G.X, G.Y, 0, GL_RGB, GL_UNSIGNED_BYTE,
               gridImage);
  glEnable(GL_TEXTURE_2D);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
  glBindTexture(GL_TEXTURE_2D, texName);
  glBegin(GL_QUADS);
  glTexCoord2f(0.0, 0.0);
  glVertex3f(0, 0, 0);
  glTexCoord2f(0.0, 1.0);
  glVertex3f(0, G.Y, 0);
  glTexCoord2f(1.0, 1.0);
  glVertex3f(G.X, G.Y, 0);
  glTexCoord2f(1.0, 0.0);
  glVertex3f(G.X, 0, 0);
  glEnd();
  glFlush();
  glDisable(GL_TEXTURE_2D);
}

#undef VERT
#undef BLANC
#undef ROUGE
#undef VERT_FONCE
#undef ORANGE
#undef FOND_NOIR

void waitGridDelay(grid G, unsigned int delay, unsigned int frame_delay) {
  const unsigned int last_tick = SDL_GetTicks();
  unsigned int current_tick = SDL_GetTicks();

  while (running && current_tick - last_tick < delay) {
    handleEvent(false);
    drawGridImage(G);
    SDL_GL_SwapWindow(window);

    if (delay - (current_tick - last_tick) > frame_delay)
      SDL_Delay(frame_delay);
    else
      SDL_Delay(delay - (current_tick - last_tick));
    current_tick = SDL_GetTicks();
  }
}

void drawGrid(grid G) {
  static unsigned int last_tick = 0;
  static unsigned int last_drawn_call = 0;
  static unsigned int call_count = 0;

  const unsigned int frame_rate = 50;
  const unsigned int frame_delay = 1000 / frame_rate;

  call_count++;

  const unsigned int current_tick = SDL_GetTicks();
  unsigned int next_drawn_call = call_count;

  if (!update) next_drawn_call = last_drawn_call + call_speed / frame_rate;

  if (next_drawn_call > call_count) return;

  unsigned int delay = 0;
  unsigned int elasped_tick = current_tick - last_tick;
  unsigned int elasped_call = call_count - last_drawn_call;

  if (elasped_call * 1000 > elasped_tick * call_speed)
    delay = elasped_call * 1000 / call_speed - elasped_tick;

  // ceci intervient quand call_speed diminue
  // le choix suivant est raisonnable au vu de la vitesse des entrées
  // utilisateur: on dessine la grille sans attendre
  if (elasped_call > call_speed) delay = 0;

  if (!update) waitGridDelay(G, delay, frame_delay);

  makeImage(&G);
  handleEvent(false);

  drawGridImage(G);

  // Affiche le résultat puis attend un certain délai
  SDL_GL_SwapWindow(window);
  last_tick = current_tick;
  last_drawn_call = call_count;
}

bool handleEvent(bool wait_event) {
  bool vertices_have_changed = false;
  SDL_Event e;

  if (wait_event)
    SDL_WaitEvent(&e);
  else if (!SDL_PollEvent(&e))
    return false;

  do {
    switch (e.type) {
      case SDL_QUIT:
        running = false;
        update = false;
        speedSet(speedMax());
        break;

      case SDL_KEYDOWN:
        if (e.key.keysym.sym == SDLK_q) {
          running = false;
          update = false;
          speedSet(speedMax());
          break;
        }
        if (e.key.keysym.sym == SDLK_p) {
          SDL_Delay(500);
          SDL_WaitEvent(&e);
          break;
        }
        if (e.key.keysym.sym == SDLK_c) {
          erase = !erase;
          break;
        }
        if (e.key.keysym.sym == SDLK_o) {
          oriented = !oriented;
          break;
        }
        if (e.key.keysym.sym == SDLK_r) {
          root = !root;
          break;
        }
        if (e.key.keysym.sym == SDLK_t) {
          mst = (mst + 1) & 3;  // +1 modulo 4
          break;
        }
        if (e.key.keysym.sym == SDLK_s) {
          sizePt *= 1.75;  // 1.75^5 = 16.41
          if (sizePt > 17.0f) sizePt = 1.0f;
          break;
        }
        if (e.key.keysym.sym == SDLK_z || e.key.keysym.sym == SDLK_KP_MINUS) {
          speedDown();
          break;
        }
        if (e.key.keysym.sym == SDLK_a || e.key.keysym.sym == SDLK_KP_PLUS) {
          speedUp();
          break;
        }
        break;
      case SDL_WINDOWEVENT:
        if (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
          double x, y;
          getCenterCoord(&x, &y);
          glViewport(0, 0, e.window.data1, e.window.data2);
          glMatrixMode(GL_PROJECTION);
          glLoadIdentity();
          glOrtho(0.0, e.window.data1, e.window.data2, 0.0f, 0.0f, 1.0f);
          glTranslatef(e.window.data1 / 2 - x, e.window.data2 / 2 - y, 0.0f);
          zoomAt(scale, x, y);
          SDL_GetWindowSize(window, &width, &height);
          SDL_SetWindowTitle(window, getTitle());
        }
        break;

      case SDL_MOUSEWHEEL:
        if (e.wheel.y > 0) zoomMouse(2.0);
        if (e.wheel.y < 0) zoomMouse(0.5);
        SDL_GetWindowSize(window, &width, &height);
        SDL_SetWindowTitle(window, getTitle());
        break;

      case SDL_MOUSEBUTTONDOWN:
        if (e.button.button == SDL_BUTTON_LEFT) {
          double x, y;
          pixelToCoord(e.motion.x, e.motion.y, &x, &y);
          if (hover) {
            int v = getClosestVertex(x, y);
            // double d=hypot(x-vertices[v].x, y-vertices[v].y);
            // printf("pixel=(%i,%i) coord=(%g,%g) vexter=%i dist=%g\n",
            //       e.motion.x,e.motion.y, x,y, v, d*d);
            if ((x - vertices[v].x) * (x - vertices[v].x) +
                    (y - vertices[v].y) * (y - vertices[v].y) <
                (sizePt * sizePt) + 2)
              selectedVertex = v;
          }
          mouse_ldown = true;
        }
        if (e.button.button == SDL_BUTTON_RIGHT) mouse_rdown = true;
        break;

      case SDL_MOUSEBUTTONUP:
        if (e.button.button == SDL_BUTTON_LEFT) {
          selectedVertex = -1;
          mouse_ldown = false;
        }
        if (e.button.button == SDL_BUTTON_RIGHT) mouse_rdown = false;
        break;

      case SDL_MOUSEMOTION:
        if (hover && !mouse_rdown && mouse_ldown && selectedVertex >= 0) {
          pixelToCoord(e.motion.x, e.motion.y, &(vertices[selectedVertex].x),
                       &(vertices[selectedVertex].y));
          vertices_have_changed = true;
        }
        if (mouse_rdown) {
          glTranslatef(e.motion.xrel / scale, e.motion.yrel / scale, 0);
        }
        break;
    }
  } while (SDL_PollEvent(&e));

  return vertices_have_changed;
}
