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

    private static Gson gson;// = new Gson();

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

    private static void initGson() {
        if (gson == null) {
            GsonBuilder builder = new GsonBuilder();
            builder.registerTypeAdapter(Product.class, new InterfaceAdapter());
            gson = builder.create();
        }
    }

    public static void save() {
        initGson();
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
        initGson();
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
