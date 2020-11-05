package com.cgoedefroit.RLE;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;

public class RLECompression implements ICompression {

    /**
     * Flacg use RLE compression
     */
    private static final char flag = '@';

    /**
     * Use it for apply RLE compression
     * @param data File stream will be compressed
     * @param out File stream will recive compressed data
     * @throws IOException Exception with file manipulation
     */
    @Override
    public void compress(Reader data, Writer out) throws IOException {
        StringBuilder result = new StringBuilder();
        int lastCCode = data.read();
        int cCode = data.read();
        while (lastCCode != -1) {
            int t = 1;
            while (cCode != -1 && lastCCode == cCode) {
                cCode = data.read();
                t++;
            }
            char c = (char) lastCCode;
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
                    result.append(String.valueOf(c).repeat(t));
                    if (c == flag) result.append(0);
                    t -= t;
                }
            lastCCode = cCode;
            cCode = data.read();
        }
        out.write(result.toString());
        out.flush();
    }

    /**
     * Use it for RLE uncompression
     * @param data File stream will be uncompressed
     * @param out File stream will recive uncompressed data
     * @throws IOException Exception with file manipulation
     * @throws RLEException Exception on file content is not valide
     */
    @Override
    public void uncompress(Reader data, Writer out) throws IOException, RLEException {
        StringBuilder result = new StringBuilder();
        int c1 = data.read();
        int c2 = data.read();
        int c3 = data.read();
        while (c1 != -1) {
            // @0 : c
            if ((char) c1 == flag) {
                if (Character.getNumericValue(c2) == 0) {
                    result.append(flag);
                    c1 = c3;
                    c2 = data.read();
                    c3 = data.read();
                } else
                    throw new RLEException("Invalide compress data string flag", result.toString(), dumpData(c1, c2, c3, data));
            } else if ((char) c1 != flag && (char) c2 != flag && c2 != -1) { // cc : cc
                result.append((char) c1);
                c1 = c2;
                c2 = c3;
                c3 = data.read();
            } else if ((char) c1 != flag && (char) c2 == flag && Character.getNumericValue(c3) == 0) { // c@0 : c@
                result.append((char) c1);
                result.append(flag);
                c1 = data.read();
                c2 = data.read();
                c3 = data.read();
            } else {
                int t = Character.getNumericValue(c3);
                if ((char) c1 != flag && (char) c2 == flag && t != 0 && Character.isDigit(c3)) { // c@3 : ccc
                    result.append(String.valueOf((char) c1).repeat(t));
                    c1 = data.read();
                    c2 = data.read();
                    c3 = data.read();
                } else if ((char) c1 != flag && c2 == -1 && !Character.isDigit(c3)) { // c : c
                    result.append((char) c1);
                    c1 = data.read();
                } else if (!Character.isDigit(c3))
                    throw new RLEException("Invalide compress data string digit", result.toString(), dumpData(c1, c2, c3, data));
                else throw new RLEException("Unknown exception", result.toString(), dumpData(c1, c2, c3, data));
            }
        }
        out.write(result.toString());
        out.flush();
    }

    private static String dumpData(int c1, int c2, int c3, Reader data) throws IOException {
        StringBuilder s = new StringBuilder();
        s.append((char) c1);
        if (c2 != -1) s.append((char) c2);
        if (c3 != -1) s.append((char) c3);
        while (true) {
            int c = data.read();
            if (c == -1) break;
            s.append((char) c);
        }
        return s.toString();
    }
}
