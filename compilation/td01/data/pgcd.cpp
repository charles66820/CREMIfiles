/////////////////////////////////////////////
// Commentaire sur une seule ligne
// compilation: c++ -o pgcd pgcd.cc
// execution: ./pgcd
/////////////////////////////////////////////

/*
 *
 * Commentaire sur plusieurs lignes
 */

#include <iostream>
using namespace std;

int main(int argn, char **argv){
  // Essai pour les nombres
  double x = 65498;
  x = 654.654;
  x = 894e-456;
  x = -654E+34;
  x = 31.416e-10;

  // Essai pour pgcd de deux entiers
  int a;
  int b;
  std::cout << "Entrer a: ";
  std::cin >> a;
  std::cout << "Entrer b: ";
  std::cin >> b;

  while(true){
    if (b > a){
      b = b % a;
      if (b==0) break;
    }
    else{
      a = a % b;
      if (a == 0) break;
    }
  }
  std::cout << "pgcd: ";
  if (a == 0)
    std::cout << b << std::endl;
  else
    std::cout << a << std::endl;

}
