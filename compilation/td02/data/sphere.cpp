/**
    CS-11 Asn 6
    sphere.cpp
    Purpose: Calculates the area of a circle and the volume
    of a sphere.
    @author Ed Parrish
    @version 1.0 3/17/04
*/

#include <iostream>

#include <cmath>
using namespace std;

const double PI = 3.14159;

/**
    Returns the area of a circle with the specified radius.

    @param radius The radius of the circle.
    @return The area of the circle.
*/
double area(double radius);

/**
    Returns the volume of a sphere with the specified radius.
 
    @param radius The radius of the circle.
    @return The volume of the sphere.
*/
double volume(double radius);

// Controls operation of the program.
int main(void) {
    double radius_of_both, area_of_circle, volume_of_sphere;

    cout << "Enter a radius to use for both a circle\n"
            << "and a sphere (in inches): ";
    cin >> radius_of_both;

    area_of_circle = area(radius_of_both);
    volume_of_sphere = volume(radius_of_both);

    cout << "Radius = " << radius_of_both << " inches\n"
            << "Area of circle = " << area_of_circle
            << " square inches\n"
            << "Volume of sphere = " << volume_of_sphere
            << " cubic inches\n";

    return 0;
}

// Returns the area of a circle with the specified radius.
double area(double radius) {
    return (PI * pow(radius, 2));
}

// Returns the volume of a sphere with the specified radius.
double volume(double radius) {
    return ((4.0 / 3.0) * PI * pow(radius, 3));
}
