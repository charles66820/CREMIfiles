package com.cgoedefroit.test;

import com.cgoedefroit.RLE.RLECompression;
import com.cgoedefroit.RLE.RLEException;

import java.io.*;

public class TestRLECompression {
    public static void main(String[] args) throws IOException {
        RLECompression bRleC = new RLECompression();

        // Create string for test compression
        String d = "onnnneee ssstttrrrrriiiiiiiiiiiiiiiiiiiiiiinnnnnnnngggggg ffffoooorrrrrrr ttttteeesssteeee avec charlesssssss@@@@@magicorp.fr";
        System.out.println("String to be compressed : " + d);

        // open input and output directories
        BufferedReader in = new BufferedReader(new FileReader("bitacu.txt"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("bitacu.min.text")));

        // Try compression
        try {
            bRleC.compress(d, out);
            System.out.println("Data compressed with succsess");
        } catch (IOException e) {
            System.out.println("Error on compress string : " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
        }
        out.println("");
        // Try uncompression
        String compressString = "o@1n@4e@3 @1s@3t@3r@5i@9i@9i@6n@8g@6 @1f@5o@4r@7 @1t@5e@3s@3t@1e@4 @1a@1v@1e@1c@1 @1c@1h@1a@1r@1l@1e@1s@7@@5m@1a@1g@1i@1c@1o@1r@1p@1.@1f@1r@1";
        try {
            bRleC.uncompress(compressString, out);
            System.out.println("Data uncompressed with succsess");
        } catch (RLEException e) {
            System.out.println("Error on uncompress string : " + e.getMessage() + "\n    decoded data : " + e.getProcessedData() + "\n    not processed data : " + e.getUnProcessedData());
        } catch (IOException e) {
            System.out.println("Error on uncompress string : " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
