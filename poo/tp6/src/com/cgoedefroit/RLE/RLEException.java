package com.cgoedefroit.RLE;

public class RLEException extends Exception {
    private final String processedData;
    private final String unProcessedData;

    /**
     * RLEException constructor
     * @param msg Exception message
     * @param processedData Processe data sting
     * @param unProcessedData Unrocesse data sting
     */
    public RLEException(String msg, String processedData, String unProcessedData) {
        super(msg);
        this.processedData = processedData;
        this.unProcessedData = unProcessedData;
    }

    /**
     * Getter for processed data
     * @return Processe data sting
     */
    public String getProcessedData() {
        return processedData;
    }

    /**
     * Getter for unprocessed data
     * @return Unrocesse data sting
     */
    public String getUnProcessedData() {
        return unProcessedData;
    }
}
