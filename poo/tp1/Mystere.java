import java.util.Random;
import java.util.Scanner;

class Mystere {

	// Permet de lire ce que l'utilisateur saisi au clavier
  public static Scanner clavier = new Scanner(System.in);

	public static void main(String[] args) {
    boolean win = false;
    Random rand = new Random();
    int mystere = rand.nextInt(100);

    int hit = 0;
    do {
      System.out.print("Nombre : ");
      int essai = clavier.nextInt();


      if (mystere == essai) win = true;
      else {
        hit++;
        System.out.println("Votre nombre " + essai + " et plus " + (mystere > essai? "petit" : "grand") + " que le nombre mystère. Vous avez fait " + hit + " sur 10");
      };
    } while(hit < 10 && !win);

    if (win) System.out.println("Vous avez trouver le nombre mystère " + mystere);
    else System.out.println("Vous avez perdu nombre mystère est " + mystere);
	}
}