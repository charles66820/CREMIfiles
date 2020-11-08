package com.cgoedefroit.td7;

import java.util.*;

public class collectionTest {
    private final static Random randomizer = new Random();

    public static ArrayList<Point2D> createRandom(int n, int range) {
        ArrayList<Point2D> pList = new ArrayList<>();

        // Add some shapes
        for (int i = 0; i < n; i++) {
            pList.add(new Point2D(
                    range * randomizer.nextDouble(),
                    range * randomizer.nextDouble(),
                    "point " + i
            ));
        }
        return pList;
    }

    public static int removeInsideDisk(List<Point2D> l, Point2D c, int r) {
        Iterator<Point2D> pIterator = l.iterator();
        int count = 0;
        while (pIterator.hasNext()) {
            Point2D p = pIterator.next();
            if (c.distance(p) <= r) {
                l.remove(p);
                count++;
            }
        }
        return count;
    }

    public static void main(String[] args) {
        // Points for test
        Point2D p1 = new Point2D(1, 2);
        Point2D p2 = new Point2D(0, 2);
        Point2D p3 = new Point2D(1, 3);
        Point2D p4 = new Point2D(0, 4);
        List<Point2D> pList;

        // Test if is possible to add multiple occurance of a point
        System.out.println("Test add multiple occurance of a point in collection :");
        pList = new ArrayList<Point2D>();
        pList.add(p1);
        pList.add(p2);
        pList.add(p1);
        for (Point2D p : pList) System.out.println(p);
        // Response : yes is possible
        pList = null;

        System.out.println();

        // Generate point list (array)
        long beginTime = System.currentTimeMillis();
        pList = createRandom(10000, 100);
        long endTime = System.currentTimeMillis();
        System.out.format("Generate point list size 10000 in range 100 (%dms)\n", endTime - beginTime);

        // Test translate all points to (-50, 50) with for index
        beginTime = System.currentTimeMillis();
        for (int i = 0; i < pList.size(); i++) pList.get(i).translate(-50, 50);
        endTime = System.currentTimeMillis();
        System.out.format("Test translate all points to (-50, 50) with for index (%dms)\n", endTime - beginTime);        // Test translate all points to (-50, 50)

        // Test translate all points to (-50, 50) with iterator
        beginTime = System.currentTimeMillis();
        Iterator<Point2D> pIterator = pList.iterator();
        while (pIterator.hasNext()) pIterator.next().translate(50, -50);
        endTime = System.currentTimeMillis();
        System.out.format("Test translate all points to (-50, 50) with iterator (%dms)\n", endTime - beginTime);        // Test translate all points to (-50, 50)

        // Test translate all points to (-50, 50) with foreach
        beginTime = System.currentTimeMillis();
        for (Point2D p : pList) p.translate(-50, 50);
        endTime = System.currentTimeMillis();
        System.out.format("Test translate all points to (-50, 50) with foreach (%dms)\n", endTime - beginTime);
        // The best version is foreach but is arraylist

        // Test edit 100000 time the middle point of list
        beginTime = System.currentTimeMillis();
        for (int i = 0; i < 100000; i++) pList.get(pList.size() / 2).translate(2);
        endTime = System.currentTimeMillis();
        System.out.format("Test edit 100000 time the middle point of list (%dms)\n", endTime - beginTime);        // Test translate all points to (-50, 50)

        System.out.println();

        // Generate point list (array) 1000000
        beginTime = System.currentTimeMillis();
        pList = createRandom(1000000, 100);
        for (Point2D p : pList) p.translate(-50, -50);
        endTime = System.currentTimeMillis();
        System.out.format("Generate point list size 1000000 in range between -50 and 50 (%dms)\n", endTime - beginTime);

        // Test removeInsideDisk on point list with 1000000 items
        beginTime = System.currentTimeMillis();
        int d = removeInsideDisk(pList, new Point2D(0, 0), 50);
        endTime = System.currentTimeMillis();
        System.out.format("Test removeInsideDisk on point list with 1000000 items. %d items deleted (%dms)\n", d, endTime - beginTime);        // Test translate all points to (-50, 50)


    }
}
