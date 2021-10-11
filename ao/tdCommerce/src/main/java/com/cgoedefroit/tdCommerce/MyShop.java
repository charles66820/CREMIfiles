package com.cgoedefroit.tdCommerce;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import com.google.gson.stream.JsonReader;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class MyShop {
    private static Set<Stock> stocks = new HashSet<>();

    private static Gson gson = new Gson();

    public MyShop() {
        GsonBuilder builder = new GsonBuilder();
        builder.registerTypeAdapter(Product.class, new InterfaceAdapter());
        Gson gson = builder.create();
    }

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

    public static void save() {
        String jsonString = gson.toJson(stocks);
        try {
            FileWriter myWriter = new FileWriter("save.json");
            myWriter.write(jsonString);
            myWriter.close();
            System.out.println("Save successfully done.");
        } catch (IOException e) {
            System.out.println("An error occurred when saving.");
        }
    }

    public static boolean load() {
        Set<Stock> tmpStocks;
        try {
            JsonReader reader = new JsonReader(new FileReader("save.json"));
            tmpStocks = gson.fromJson(reader, new TypeToken<Set<Stock>>(){}.getType());
        } catch (IOException e) {
            System.out.println("An error occurred when we load the save file.");
            return false;
        }
        if (tmpStocks == null) return false;
        stocks = tmpStocks;
        return true;
    }
}
