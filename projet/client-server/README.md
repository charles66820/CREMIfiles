# Client-sever

## Compte rendu

### Backend

J'ai implémenter :

- La class `ImageDao`
- Les route `GET /images/{id}`, `POST /images`, `DELETE /images/{id}` et `GET /images`.
  J'ai respecter le http status code pour chaque route.
  J'ai aussi pris la liberté pour la route `POST /images` d'envoyer en réponse l'id de la nouvel image créer avec le `Content-Type` `application/json;charset=UTF-8`.
- Les test JUnit avec Mock. Pour c'est test j'ai ajouter les verifications suivante :
  - Si le status est le bon.
  - Pour la route `GET /images` : Si le contenu de la réponse est le bon.
  - La verification du `Content-Type` dans le `header` le la réponse.
  - Pour le test `deleteImageShouldReturnBadRequest` :
    Si le serveur retourne bien le http status code `400` quant l'id est du text ou qu'il est trop grans pour être un long.

Je n'ai eu aucune difficulté à implémenter ces fonctionnalités.

### Frontend

Comme vous pouvez le voir j'ai bien ajouter et modifier tous les `pom.xml`.

J'ai renommé le composent `HelloWorld.vue` en `Image.vue`.

J'ai implémenter :

- La liste déroulante qui permet d'afficher l'image sélectionnée.
- Le formulaire pour déposer une image sur le serveur.
- Un nouveau composant nommé `Images.vue` qui afficher l'ensemble des images présentes sur le serveur sous forme d'une galerie.
- La factorisation des appelés au serveur dans `http-api.js`.

À implémenter mais Hors-barème :

- Les tests unitaires du client
- Les tests d'intégration / fonctionnels (End-2-End) avec Nightwatch.

## Run and dev

Install

```bash
mvn clean install
```

Run production server

```bash
mvn --projects backend spring-boot:run
```

Run dev frontend server

```bash
npm run serve
```
