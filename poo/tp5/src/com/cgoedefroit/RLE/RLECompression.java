package com.cgoedefroit.RLE;

public class RLECompression implements ICompression {

    private static final char flag = '@';

    private int lengthOfSingleLetterPrefix(String s) {
        char c = s.charAt(0);
        int i = 1;
        while (i < s.length() && s.charAt(i) == c) i++;
        return i;
    }

    @Override
    public String compress(String data) throws Exception {
        StringBuilder result = new StringBuilder("");
        while (data.length() != 0) {
            char c = data.charAt(0);
            int L = lengthOfSingleLetterPrefix(data);
            int t = L;
            while (t != 0)
                if (t > 9) {
                    if (c == flag) for (int i = 0; i < t; i++) result.append(flag).append(0);
                    else result.append(c).append(flag).append('9');
                    t -= 9;
                } else if (t > 3) {
                    if (c == flag) for (int i = 0; i < t; i++) result.append(flag).append(0);
                    else result.append(c).append(flag).append(t);
                    t -= t;
                } else {
                    for (int i = 0; i < t; i++) result.append(c);
                    if (c == flag) result.append(0);
                    t -= t;
                }
            data = data.substring(L);
        }
        return result.toString();
    }

    @Override
    public String uncompress(String data) throws Exception {
        StringBuilder result = new StringBuilder();
        while (data.length() != 0) {
            if (data.length() < 3)
                throw new RLECompressionException("Invalide compress data string length", result.toString(), data);
            if (data.charAt(1) != flag)
                throw new RLECompressionException("Invalide compress data string flag", result.toString(), data);
            char c = data.charAt(0);
            char L = data.charAt(2);
            int t = Integer.parseInt("" + L);
            for (int i = 0; i < t; i++) result.append(c);
            data = data.substring(3);
        }
        return result.toString();
    }
}
