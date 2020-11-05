package com.cgoedefroit.td3_4.shape.body;

import com.cgoedefroit.td3_4.shape.Shape2D;
import com.cgoedefroit.td3_4.shape.elementary.Point2D;

import java.util.ArrayList;

public abstract class Polygone extends Shape2D {

    // Methods
    public abstract ArrayList<Point2D> vertices();
}
