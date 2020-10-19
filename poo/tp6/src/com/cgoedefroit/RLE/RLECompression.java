package com.cgoedefroit.RLE;

import java.io.IOException;
import java.io.Writer;

public class RLECompression implements ICompression {

    private static final char flag = '@';

    private int lengthOfSingleLetterPrefix(String s) {
        char c = s.charAt(0);
        int i = 1;
        while (i < s.length() && s.charAt(i) == c) i++;
        return i;
    }

    @Override
    public void compress(String data, Writer out) throws IOException {
        StringBuilder result = new StringBuilder("");
        while (data.length() != 0) {
            char c = data.charAt(0);
            int L = lengthOfSingleLetterPrefix(data);
            int t = L;
            while (t != 0)
                if (t > 9) {
                    if (c != flag) result.append(c);
                    result.append(flag).append('9');
                    t -= 9;
                } else if (t > 3) {
                    if (c != flag) result.append(c);
                    result.append(flag).append(t);
                    t -= t;
                } else {
                    result.append(c);
                    t -= t;
                }
            data = data.substring(L);
        }
        out.write(result.toString());
        out.flush();
    }

    @Override
    public void uncompress(String data, Writer out) throws IOException, RLEException {
        StringBuilder result = new StringBuilder();
        while (data.length() != 0) {
            if (data.length() < 3) throw new RLEException("Invalide compress data string length", result.toString(), data);
            if (data.charAt(1) != flag) throw new RLEException("Invalide compress data string flag", result.toString(), data);
            char c = data.charAt(0);
            char L = data.charAt(2);
            int t = Integer.parseInt("" + L);
            result.append(String.valueOf(c).repeat(t));
            data = data.substring(3);
        }
        out.write(result.toString());
        out.flush();
    }
}
