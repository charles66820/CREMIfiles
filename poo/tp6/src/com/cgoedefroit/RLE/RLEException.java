package com.cgoedefroit.RLE;

public class RLEException extends Exception {
    private String processedData;
    private String unProcessedData;

    public RLEException(String msg, String processedData, String unProcessedData) {
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
