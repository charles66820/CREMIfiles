package com.cgoedefroit.tdDdd.exception;

public class IntValidationException extends ValidationException {

    private final int minValue;
    private final int maxValue;

    public IntValidationException(String s, int minValue, int maxValue) {
        super(s);
        this.minValue = minValue;
        this.maxValue = maxValue;
    }

    public IntValidationException(String s, int minValue) {
        this(s, minValue, Integer.MAX_VALUE);
    }

    public IntValidationException(String s) {
        this(s, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }

    public int getMinValue() {
        return minValue;
    }

    public int getMaxValue() {
        return maxValue;
    }

}
