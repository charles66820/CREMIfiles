package com.cgoedefroit.BasicRLE;

public interface ICompression {
    public String compress(String data) throws Exception;
    public String uncompress(String data) throws Exception;
}
