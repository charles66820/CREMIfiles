
/*@ predicate inInterval(integer a, integer b, integer c) = a >= b && b <= c;

    predicate fourInOrder(integer a, integer b, integer c, integer d) = inInterval(b,a,c) && inInterval(c,b,d);

    predicate controlledEquality(integer a, integer b, boolean c) = a == b || c;
*/

/*@ requires inInterval(a,b,c);
    requires inInterval(c,a,d);
    ensures fourInOrder(\result,a,c,d);
*/
int toto(int a, int b, int c, int d)
{
    return b;
}