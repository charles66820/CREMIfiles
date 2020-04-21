<?php

// file_get_contents($filename) voir pour un csv

$str = "Minecraft;Pokemon;Zelda;The Witcher;Kerbal space program";
$t = explode(";", $str);

$text = $t[mt_rand(0,  count($t))];
?>

<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8"><!-- Permet d'utiliser des caracter spesieau  -->

  <!-- Les mots cler pour le referancement de la page. permet aux robots d'indexsation de savoire de quoi la page parle-->
  <meta name="keywords" content="HTML,CSS,XML,JavaScript,demo,inutile">

  <!-- Pour mettre ca signature -->
  <meta name="author" content="Charles Goedefroit">

  <!-- Pour que l'echelle de la page s'adapte a la taillede l'ecran -->
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <!-- Permet de metre un description a la page pour les robots d'indexsation -->
  <meta name="description" content="">

  <!--  -->
  <link rel="icon" href="img/icon.gif" type="image/gif">

  <link rel="stylesheet" href="css/main.css">
  <link rel="stylesheet" href="css/home.css">
  <title>Document HTML pour le TP5</title>
</head>

<body>
  <header>
    <nav id="li">
      <a class="nav-item icon" href="index.html">
        <img src="img/icon.gif" alt="site icon" height="60">
      </a>
      <a class="nav-item" href="index.html">Home</a>
      <a class="nav-item active" href="page.php">Page PHP</a>
    </nav>
  </header>
  <main>
    <article>
      <h1>Le titre de la premier section</h1>
      <section>
        <h2>Le titre de la premier sous section</h2>
        <p>Du text pour la premier sous section</p>
        <p>Text aleatoire selectionner par le PHP : <?= $text ?></p>
      </section>
      <section>
        <h2>Le titre de la desieme sous section</h2>
        <p>Du text pour la desieme sous section</p>
      </section>
    </article>
    <article>
      <h1>Le titre de la desieme section</h1>
      <section>
        <h2>Le titre de la premier sous section</h2>
        <p>Du text pour la premier sous section</p>
      </section>
    </article>
    <article>
      <h1>Liens</h1>
      <section>
        <h2>Actions</h2>
        <a href="#top">Aller en haut de la page</a>
      </section>
    </article>
  </main>
  <footer>
    <a href="index.html">Home</a>
    <a href="page.php">Page PHP</a>
  </footer>
  <script src="js/home.js"></script>
</body>

</html>