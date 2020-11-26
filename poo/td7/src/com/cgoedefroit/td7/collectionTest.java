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
        List<Point2D> rmList = new ArrayList<>();
        for (Point2D p : l) if (c.distance(p) <= r) rmList.add(p);
        for (Point2D point2D : rmList) l.remove(point2D);
        return rmList.size();
    }

    private static List<Point2D> createInterlacingListIndexed(List<Point2D> pList, List<Point2D> pList2) {
        List<Point2D> ret = new ArrayList<>();
        int lenMin = Math.min(pList.size(), pList2.size()); // n = little list
        List<Point2D> bigestPointList = pList.size() < pList2.size() ? pList2 : pList;
        for (int i = 0; i < lenMin * 2; i++) {
            if (i % 2 == 0) {
                int j = (i / 2);
                if (j < pList.size()) ret.add(pList.get(j));
                else ret.add(pList2.get(j));
            } else {
                int j = (i / 2);
                if (j < pList2.size()) ret.add(pList2.get(j));
                else ret.add(pList.get(j));
            }
        }
        for (int j = lenMin; j < bigestPointList.size(); j++) ret.add(bigestPointList.get(j));
        return ret;
    }

    private static List<Point2D> createInterlacingListIterat(List<Point2D> pList, List<Point2D> pList2) {
        List<Point2D> ret = new ArrayList<>();
        Iterator<Point2D> it = pList.iterator();
        Iterator<Point2D> it2 = pList2.iterator();
        for (int i = 0; i < pList.size() + pList2.size(); i++) {
            if (i % 2 == 0)
                if (it.hasNext()) ret.add(it.next());
                else ret.add(it2.next());
            else if (it2.hasNext()) ret.add(it2.next());
            else ret.add(it.next());
        }
        return ret;
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
        pList = createRandom(100000, 100);
        for (Point2D p : pList) p.translate(-50, -50);
        endTime = System.currentTimeMillis();
        System.out.format("Generate point list size 1000000 in range between -50 and 50 (%dms)\n", endTime - beginTime);

        // Test removeInsideDisk on point list with 1000000 items
        beginTime = System.currentTimeMillis();
        int d = removeInsideDisk(pList, new Point2D(0, 0), 50);
        endTime = System.currentTimeMillis();
        System.out.format("Test removeInsideDisk on point list with 1000000 items. %d items deleted (%dms)\n", d, endTime - beginTime);        // Test translate all points to (-50, 50)

        // Two methods interlacing
        pList = createRandom(1000000, 100);
        List<Point2D> pList2 = createRandom(1000000, 100);

        beginTime = System.currentTimeMillis();
        List<Point2D> interlacingPointList1 = createInterlacingListIndexed(pList, pList2);
        endTime = System.currentTimeMillis();
        System.out.format("Create interlacing list from two list with index. (%dms)\n", endTime - beginTime);

        beginTime = System.currentTimeMillis();
        List<Point2D> interlacingPointList2 = createInterlacingListIterat(pList, pList2);
        endTime = System.currentTimeMillis();
        System.out.format("Create interlacing list from two list with iterator. (%dms)\n", endTime - beginTime);

        // Sort list
        beginTime = System.currentTimeMillis();
        interlacingPointList2.sort(new Comparator<Point2D>() {
            @Override
            public int compare(Point2D p, Point2D p1) {
                return 0;
            }
        });
        endTime = System.currentTimeMillis();
        System.out.format("Sort list. (%dms)\n", endTime - beginTime);
    }
}
