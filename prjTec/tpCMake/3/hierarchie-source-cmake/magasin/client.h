#ifndef CLIENT_H
#define CLIENT_H

typedef struct client *client;

extern client client_creer(char *nom, int nombre_d_articles);
extern void client_liberer(client self);
extern int client_nombre_d_article(client self);
extern char *client_nom(client self);
extern void client_decrementer_nombre_d_articles(client self);
extern void client_afficher(client self);

#endif  /* CLIENT_H */
