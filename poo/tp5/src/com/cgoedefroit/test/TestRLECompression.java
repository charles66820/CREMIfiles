package com.cgoedefroit.test;

import com.cgoedefroit.RLE.RLECompression;
import com.cgoedefroit.RLE.RLECompressionException;

public class TestRLECompression {
    public static void main(String[] args) {
        RLECompression bRleC = new RLECompression();

        // Create string for test compression
        String d = "onnnneee ssstttrrrrriiiiiiiiiiiiiiiiiiiiiiinnnnnnnngggggg ffffoooorrrrrrr ttttteeesssteeee avec charlesssssss@@@@@magicorp.fr";
        System.out.println("String to be compressed : " + d);

        // Try compression
        String compressString = "";
        try {
            compressString = bRleC.compress(d);
            System.out.println("String compressed : " + compressString);
        } catch (RLECompressionException e) {
            System.out.println("Error on compress string : " + e.getMessage() + "\n    encoded data : " + e.getProcessedData() + "\n    not processed data : " + e.getUnProcessedData());
        } catch (Exception e) {
            e.printStackTrace();
        }
        // Try uncompression
        String uncompressString;
        try {
            uncompressString = bRleC.uncompress(compressString);
            System.out.println("String uncompressed : " + uncompressString);
        } catch (RLECompressionException e) {
            System.out.println("Error on uncompress string : " + e.getMessage() + "\n    decoded data : " + e.getProcessedData() + "\n    not processed data : " + e.getUnProcessedData());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
