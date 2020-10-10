package com.cgoedefroit.RLE;

public class RLECompressionException extends Exception {
    private String processedData;
    private String unProcessedData;

    public RLECompressionException(String msg, String processedData, String unProcessedData) {
        super(msg);
        this.processedData = processedData;
        this.unProcessedData = unProcessedData;
    }

    public String getProcessedData() {
        return processedData;
    }

    public String getUnProcessedData() {
        return unProcessedData;
    }
}
