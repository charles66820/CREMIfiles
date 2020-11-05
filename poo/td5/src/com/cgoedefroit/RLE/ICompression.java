package com.cgoedefroit.RLE;

public interface ICompression {
    String compress(String data) throws Exception;
    String uncompress(String data) throws Exception;
}
