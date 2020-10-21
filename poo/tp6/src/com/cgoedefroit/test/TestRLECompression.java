package com.cgoedefroit.test;

import com.cgoedefroit.RLE.RLECompression;
import com.cgoedefroit.RLE.RLEException;

import java.io.*;

public class TestRLECompression {
    public static void main(String[] args) throws IOException {
        RLECompression bRleC = new RLECompression();

        BufferedReader in;
        PrintWriter out;

        // Try compression
        try {
            in = new BufferedReader(new FileReader("test.txt.in"));
            out = new PrintWriter(new BufferedWriter(new FileWriter("test.min.txt.out")));
            bRleC.compress(in, out);
            System.out.println("Data compressed with succsess");
            in.close();
            out.close();
        } catch (IOException e) {
            System.out.println("Error on compress string : " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Try uncompression
        try {
            in = new BufferedReader(new FileReader("test.min.txt.in"));
            out = new PrintWriter(new BufferedWriter(new FileWriter("test.txt.out")));
            bRleC.uncompress(in, out);
            System.out.println("Data uncompressed with succsess");
            in.close();
            out.close();
        } catch (RLEException e) {
            System.out.println("Error on uncompress string : " + e.getMessage() + "\n    decoded data : " + e.getProcessedData() + "\n    not processed data : " + e.getUnProcessedData());
        } catch (IOException e) {
            System.out.println("Error on uncompress string : " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
