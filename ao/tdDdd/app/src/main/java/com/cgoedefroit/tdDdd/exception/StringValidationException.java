package com.cgoedefroit.tdDdd.exception;

public class StringValidationException extends ValidationException {
    private final int minLength;
    private final int maxLength;
    private final boolean isAlpha;

    public StringValidationException(String s, int minLength, int maxLength, boolean isAlpha) {
        super(s);
        this.minLength = minLength;
        this.maxLength = maxLength;
        this.isAlpha = isAlpha;
    }

    public StringValidationException(String s) {
        this(s, 0, 0, false);
    }

    public int getMinLength() {
        return minLength;
    }

    public int getMaxLength() {
        return maxLength;
    }

    public boolean isAlpha() {
        return isAlpha;
    }
}
