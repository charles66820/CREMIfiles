package com.cgoedefroit.tdCommerce;

import java.util.*;

public class MyShop {
    private static final Set<Stock> stocks = new HashSet<>();

    static boolean addStock(String name, String address) {
        return stocks.add(new Stock(name, address));
    }

    static Stock selectAStock(String name) {
        for (Stock stock : stocks)
            if (Objects.equals(stock.getName(), name))
                return stock;
        return null;
    }

    static boolean hasStock() {
        return stocks.size() != 0;
    }

    public static void printStocks() {
        stocks.forEach(stock -> System.out.println(" * " + stock.toString()));
    }
}
