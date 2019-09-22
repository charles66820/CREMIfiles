#include "tsp.h"
#include "memoire.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <assert.h>

#define DIM "DIMENSION:"
#define EDGE "EDGE_WEIGHT_SECTION"


#define MAX_DOUBLE 10000000

coordonnees *villes = NULL;

struct chemin 
{
    int *tab;
    int taille;
};

chemin chemin_creer(int taille) {
  chemin self = memoire_allouer(sizeof(struct chemin));
  self->taille = taille;
  self->tab = memoire_allouer(sizeof(int)*taille);
  return self;
}

void chemin_liberer(chemin self) {
  memoire_liberer(self->tab);
  memoire_liberer(self);
}

void chemin_copier (chemin src, chemin dst) {
  assert(src->taille == dst->taille);
  for (int i = 0; i < src->taille;  i++)
    dst->tab[i] = src->tab[i];
}

char * chercher_lire_ligne_commencant_par(char * mot, FILE *fichier)
{
  enum{Taille_ligne = 1024};
  char *ligne=memoire_allouer(Taille_ligne);
  int l = strlen(mot);
  while(fgets(ligne,Taille_ligne, fichier))
    if(!strncmp(mot,ligne,l))
      return ligne;
  free(ligne);
  return NULL;
}

coordonnees* lire_villes(char *nom_fichier, int *nb_villes) {

   FILE *f=fopen(nom_fichier,"r");
   if (f==NULL)
     {
       perror(nom_fichier);
       exit(1);
     }
   char *l=chercher_lire_ligne_commencant_par(DIM,f);
 
   sscanf(l,DIM " %d", nb_villes);
   villes= memoire_allouer(sizeof(coordonnees)* *nb_villes);
   free(l);
   l=chercher_lire_ligne_commencant_par("NODE_COORD_SECTION",f);  
   if (l==NULL)
     l=chercher_lire_ligne_commencant_par("DISPLAY_DATA_SECTION",f);  
   free(l);
   
   for(int i = 0; i < *nb_villes; i++)
     {
       int num_ville;
       fscanf(f," %d %lf %lf \n",&num_ville,&villes[i].x,&villes[i].y);
     }

   fclose(f);
   return villes;
}


double norme_L2(coordonnees v1, coordonnees v2) {
  return sqrt((v1.x-v2.x)*(v1.x-v2.x) + (v1.y-v2.y)*(v1.y-v2.y));
}

double norme_Linf(coordonnees v1, coordonnees v2) {
  return abs(v1.x-v2.x)+abs(v1.y-v2.y);
}

matrice2d charger_villes_norme(char * nom_fichier, double (*norme)(coordonnees, coordonnees)) {
  int nb_villes;
  coordonnees *coords;
  coords = lire_villes(nom_fichier, &nb_villes);
  matrice2d dist = m2d_creer(nb_villes);

  for (int i = 0; i < nb_villes; i++)
    for (int j = 0; j < nb_villes; j++) {
      coordonnees v1 = coords[i];
      coordonnees v2  = coords[j];
      double val = (*norme)(v1,v2);
      m2d_ecrire(dist,i,j, val);
    }
  memoire_liberer(coords);
  return dist;
}

matrice2d charger_villes_L2(char * nom_fichier) {
  return charger_villes_norme(nom_fichier, norme_L2);
}

matrice2d charger_villes_Linf(char * nom_fichier) {
  return charger_villes_norme(nom_fichier, norme_Linf);
}

chemin lire_tour(char *nom_fichier, int nb_villes)
{
  FILE *f=fopen(nom_fichier,"r");
  if (f==NULL)
    {
      perror(nom_fichier);
      exit(1);
    }
  char *l=chercher_lire_ligne_commencant_par("TOUR_SECTION",f);
  free(l);
  
  chemin tour = chemin_creer(nb_villes);
  
   for(int i = 0; i < nb_villes; i++)
     {
       fscanf(f," %d ",&tour->tab[i]);
       tour->tab[i]--;
     }
   
   fclose(f);
   return tour;
}

void sauver_tsplib(char * nom_fichier, matrice2d m) {
  FILE *f=fopen(nom_fichier,"w");
  int nb_villes = m2d_taille(m);
  if (f==NULL)
    {
      perror(nom_fichier);
      exit(1);
    }
  
  fprintf(f, "DIMENSION: %d\n", nb_villes);
  fprintf(f, "EDGE_WEIGHT_FORMAT: FULL_MATRIX\n");
  fprintf(f,"DISPLAY_DATA_TYPE: TWOD_DISPLAY\n");
  fprintf(f,"EDGE_WEIGHT_SECTION\n");
  for(int i = 0; i < nb_villes; i++)
    {
      for(int j = 0; j < nb_villes; j++) 
	{
	  fprintf(f,"%g ", m2d_val(m, i,j));
	}
      fprintf(f,"\n");
    }
  fclose(f);
}

matrice2d charger_tsplib(char * nom_fichier) {
  int nb_villes;
  FILE *f=fopen(nom_fichier,"r");
   if (f==NULL)
     {
       perror(nom_fichier);
       exit(1);
     }
   char *l=chercher_lire_ligne_commencant_par(DIM,f);
 
   sscanf(l,DIM " %d", &nb_villes);
   matrice2d dist = m2d_creer(nb_villes);

   chercher_lire_ligne_commencant_par(EDGE,f);
   for(int i = 0; i < nb_villes; i++)
     {
      for(int j = 0; j < nb_villes; j++) 
      {
	double val;
       if (!fscanf(f, " %lf", &val)) 
           break;
       m2d_ecrire(dist, i,  j, val);
       //       printf("%lf \n",m2d_val(dist,i,j));
      }

  }
  fclose(f);
  return dist;
}



chemin nearest_neighbor_tsp(matrice2d dist) {
  int taille = m2d_taille(dist);

  chemin tour = chemin_creer(m2d_taille(dist));

  int * visite = memoire_allouer(sizeof(int)*taille);
  for (int i = 0; i < taille; i++)
    visite[i] = 0;

  int suivant = 1;
  tour->tab[0] = 0;
  visite[0] = 1;
  while (suivant < taille) {
    int plus_proche = -1;
    int dist_plus_proche = MAX_DOUBLE;
    for (int j = 0; j < taille; j++) {
      if (j != tour->tab[suivant-1] && visite[j]==0) {
	if (dist_plus_proche > m2d_val(dist,tour->tab[suivant-1],j)) {
	  plus_proche = j;
	  dist_plus_proche = m2d_val(dist,tour->tab[suivant-1],j);
	}
      }
    }
    tour->tab[suivant] = plus_proche;
    visite[plus_proche] = 1;
    suivant++;
  } 
  return tour;
}


// http://fr.wikipedia.org/wiki/Algorithme_de_Prim
arete* calcul_mst(matrice2d distances) {
  int taille = m2d_taille(distances);
  int * visite = memoire_allouer(sizeof(int)*taille);
  arete * aretes = memoire_allouer(sizeof(arete)*taille);
  for (int i = 0; i < taille; i++)
    visite[i] = 0;
  visite[0] = 1;
  int suivant = 1;
  while (suivant <= taille) 
  {
    arete prochaine_arete;
    int dist_plus_proche = MAX_DOUBLE;
    for (int i = 0; i < taille; i++) 
    {
      for (int j = 0; j < taille; j++) 
      {
	       if (visite[i] == 1 && visite[j]==0)
	         if (dist_plus_proche > m2d_val(distances, i, j)) 
           {
	           prochaine_arete.s = i;
	           prochaine_arete.t = j;
	           dist_plus_proche = m2d_val(distances, i, j);
	         }
      }
    }
    printf("%d %d\n", prochaine_arete.s, prochaine_arete.t);
    aretes[suivant-1]=prochaine_arete;
    visite[prochaine_arete.t] = 1;
    suivant++;
  }
  memoire_liberer(visite);
  return aretes;
}


void tour_mst_recur(arete *mst, chemin tour, int *visite, int *suivant) 
{
  int s = tour->tab[*suivant-1];

  for (int i = 0; i < tour->taille; i++) 
  {
    if (mst[i].s == s && visite[mst[i].t]==0) 
    {
      tour->tab[*suivant] = mst[i].t;
      visite[mst[i].t] = 1;
	    (*suivant)++;
	    tour_mst_recur(mst, tour, visite, suivant);
    }
  }
}

chemin tour_mst(arete *mst, int taille) 
{
  chemin tour = chemin_creer(taille);
  int * visite = memoire_allouer(sizeof(int)*taille);
  for (int i = 0; i < taille; i++)
    visite[i] = 0;
  visite[0] = 1;
  int suivant = 1;
  tour->tab[0] = 0;
  tour_mst_recur(mst, tour, visite, &suivant);

  return tour;
}

chemin mst_tsp(matrice2d distances) {
  arete * mst = calcul_mst(distances);
  chemin tour = tour_mst(mst, m2d_taille(distances));
  memoire_liberer(mst);
  return tour;
}

void exhaustive_tsp_recur(matrice2d dist, chemin tour, int *visite, int suivant, double *meilleur, chemin meilleur_tour) {
  if (suivant == tour->taille) {
    if (chemin_longueur(tour, dist) < *meilleur) {
      *meilleur = chemin_longueur(tour, dist);
      chemin_copier(tour, meilleur_tour);
      printf("meilleur actu %lf\n",*meilleur);
    }
  }
  else {
    for (int i = 0; i < tour->taille; i++) {
      if (visite[i] == 0) {
	visite[i] = 1;
	tour->tab[suivant] = i;
	suivant++;
	exhaustive_tsp_recur(dist, tour, visite, suivant, meilleur, meilleur_tour);
	suivant--;
	tour->tab[suivant] = -1;
	visite[i] = 0;
      }
    }
  }
}

void exhaustive_tsp_BB_recur(matrice2d dist, chemin tour, double long_tour, int *visite, int suivant, double *meilleur, chemin meilleur_tour) {
  if (suivant == tour->taille) {
    if (chemin_longueur(tour, dist) < *meilleur) {
      *meilleur = chemin_longueur(tour, dist);
      chemin_copier(tour, meilleur_tour);
      printf("meilleur BB actu %lf\n",*meilleur);
    }
  }
  else {
    for (int i = 0; i < tour->taille; i++) {
      if (visite[i] == 0) {
	visite[i] = 1;
	tour->tab[suivant] = i;
	long_tour += m2d_val(dist,tour->tab[suivant-1],tour->tab[suivant]);
	suivant++;
	// on teste si la longueur du début du tour + la distance pour revenir à l'origine est plus petite que le meilleur tour)
	if (long_tour + m2d_val(dist,i,0) <= *meilleur)
	  exhaustive_tsp_BB_recur(dist, tour, long_tour, visite, suivant, meilleur, meilleur_tour);
	suivant--;
	long_tour -= m2d_val(dist,tour->tab[suivant-1],tour->tab[suivant]);
	tour->tab[suivant] = -1;
	visite[i] = 0;
      }
    }
  }
}


// BB = 0 : classique
// BB = 1 : Branch and Bound
chemin exhaustive_tsp_tous(matrice2d distances, int BB) {
  int taille = m2d_taille(distances);
  chemin tour = chemin_creer(taille);
  chemin meilleur_tour;
  int * visite = memoire_allouer(sizeof(int)*taille);
  for (int i = 0; i < taille; i++)
  visite[i] = 0;
  visite[0] = 1;
  int suivant = 1;
  tour->tab[0] = 0;
  double meilleur_longueur;
  if(BB == 1) {
    meilleur_tour  = mst_tsp( distances);
    meilleur_longueur= chemin_longueur(meilleur_tour, distances);
    exhaustive_tsp_BB_recur(distances, tour, 0, visite, suivant, &meilleur_longueur, meilleur_tour);
  }
  else {
    meilleur_tour = chemin_creer(taille);
    meilleur_longueur= MAX_DOUBLE;
    exhaustive_tsp_recur(distances, tour, visite, suivant, &meilleur_longueur, meilleur_tour);
    }
  chemin_liberer(tour);
  return meilleur_tour;
}

chemin exhaustive_tsp(matrice2d distances) {
  return exhaustive_tsp_tous(distances, 0);
}

chemin exhaustive_BB_tsp(matrice2d distances) {
  return exhaustive_tsp_tous(distances, 1);
}


double chemin_longueur(chemin tour, matrice2d dist) {
  double longueur = 0;
  int taille = m2d_taille(dist);
  for (int i = 1; i < taille; i++) {
    longueur+= m2d_val(dist,tour->tab[i-1],tour->tab[i]);
  }
    longueur+= m2d_val(dist,tour->tab[taille-1],tour->tab[0]);
  return longueur;
}


void chemin_afficher (chemin tour) {
  for (int i = 0 ; i < tour->taille; i++) {
    printf("%d\n",tour->tab[i]);
  }
}

int chemin_element (chemin tour, int position)
{
  return tour->tab[position];
}
