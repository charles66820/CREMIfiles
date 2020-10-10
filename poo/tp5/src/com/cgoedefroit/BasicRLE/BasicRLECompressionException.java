package com.cgoedefroit.BasicRLE;

public class BasicRLECompressionException extends Exception {
    private String processedData;
    private String unProcessedData;

    public BasicRLECompressionException(String msg, String processedData, String unProcessedData) {
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
