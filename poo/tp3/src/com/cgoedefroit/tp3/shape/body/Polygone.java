package com.cgoedefroit.tp3.shape.body;


import com.cgoedefroit.tp3.shape.Shape2D;
import com.cgoedefroit.tp3.shape.elementary.Point2D;

import java.util.ArrayList;

public abstract class Polygone extends Shape2D {

    // Methods
    public abstract ArrayList<Point2D> vertices();
}
