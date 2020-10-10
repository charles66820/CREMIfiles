package com.cgoedefroit.test;

import com.cgoedefroit.BasicRLE.BasicRLECompression;
import com.cgoedefroit.BasicRLE.BasicRLECompressionException;

public class TestBasicRLE {
    public static void main(String[] args) {
        BasicRLECompression bRleC = new BasicRLECompression();

        // Create string for test compression
        String d = "onnnneee ssstttrrrrriiiiiiiiiiiiiiiiiiiiiiinnnnnnnngggggg ffffoooorrrrrrr ttttteeesssteeee";
        System.out.println("String to be compressed : " + d);

        // Try compression
        String compressString = "";
        try {
            compressString = bRleC.compress(d);
            System.out.println("String compressed : " + compressString);
        } catch (BasicRLECompressionException e) {
            System.out.println("Error on compress string : " + e.getMessage() + "\n    encoded data : " + e.getProcessedData() + "\n    not processed data : " + e.getUnProcessedData());
        } catch (Exception e) {
            e.printStackTrace();
        }
        // Try uncompression
        String uncompressString;
        try {
            uncompressString = bRleC.uncompress(compressString);
            System.out.println("String uncompressed : " + uncompressString);
        } catch (BasicRLECompressionException e) {
            System.out.println("Error on uncompress string : " + e.getMessage() + "\n    decoded data : " + e.getProcessedData() + "\n    not processed data : " + e.getUnProcessedData());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
