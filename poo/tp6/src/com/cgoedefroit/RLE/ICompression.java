package com.cgoedefroit.RLE;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;

interface ICompression {
    public void compress(Reader data, Writer out) throws IOException;
    public void uncompress(Reader data, Writer out) throws IOException, RLEException;
}