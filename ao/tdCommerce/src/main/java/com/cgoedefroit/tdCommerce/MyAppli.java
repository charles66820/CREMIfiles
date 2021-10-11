package com.cgoedefroit.tdCommerce;

import com.google.gson.Gson;

import java.time.LocalDate;
import java.time.format.DateTimeParseException;
import java.util.Scanner;

public class MyAppli {
    private static Scanner scan;

    public static void main(String[] args) {
        System.out.println("Welcome in MyShop");
        scan = new Scanner(System.in);

        System.out.println("Don't load save ?");
        String input = scan.nextLine();
        if (!input.matches("^(?:1|t(?:rue)?|y(?:es)?|ok(?:ay)?)$"))
            MyShop.load();

        Stock currentStock;
        while (true) {
            System.out.println("Choose your action (c)reate, (s)how, (e)dit product quantitiy, (q)uit");
            input = scan.nextLine();
            switch (input) {
                case "c":
                    System.out.println("Choose what you want to create (s)tock, (p)roduct");
                    input = scan.nextLine();
                    switch (input) {
                        case "s":
                            System.out.println("Create a stock");
                            String stockName = getUserInputString("Enter a name : ");
                            String address = getUserInputString("Enter an address : ");
                            if (MyShop.addStock(stockName, address))
                                System.out.println("New stock created with success");
                            else System.out.println("A stock named \"" + stockName + "\" alrady exist");
                            break;
                        case "p":
                            if (!MyShop.hasStock()) {
                                System.out.println("You need to create stock first");
                                break;
                            }
                            System.out.println("Create a product");
                            do {
                                System.out.print("Enter the stock name where the product will be create : ");
                                input = scan.nextLine();
                            } while ((currentStock = MyShop.selectAStock(input)) == null);

                            String productName = getUserInputString("Enter a name : ");
                            int quantity = getUserInputInt("Enter a quantity : ");

                            boolean created = false;

                            System.out.println("Choose what kind of product you want to create : (s)anitary, (f)ood");
                            input = scan.nextLine();
                            switch (input) {
                                case "s":
                                    created = currentStock.addSanitaryProduct(productName, quantity);
                                break;
                                case "f":
                                    String date;
                                    LocalDate expirationDate;
                                    while (true) {
                                        date = getUserInputString("Enter a product expiration date : ");
                                        try {
                                            expirationDate = LocalDate.parse(date);
                                            if (expirationDate != null) break;
                                        } catch (DateTimeParseException e) {
                                            System.out.println("Bad date \"" + e.getMessage() + '"');
                                        }
                                    }
                                    created = currentStock.addFoodProduct(productName, quantity, expirationDate);
                                    break;
                            }

                            if (created)
                                System.out.println("New product created with success in stock \"" + currentStock.getName() + "\"");
                            else
                                System.out.println("A product named \"" + productName + "\" alrady exist in stock \"" + currentStock.getName() + "\"");
                            break;
                        default:
                            System.out.println("Creation cancel");
                    }
                    break;
                case "s":
                    System.out.println("Choose what you want to show (s)tocks, (sp) stock product, (p)roduct");
                    input = scan.nextLine();
                    switch (input) {
                        case "s":
                            if (!MyShop.hasStock()) {
                                System.out.println("No stock found");
                                break;
                            }
                            System.out.println("All stocks :");
                            MyShop.printStocks();
                            break;
                        case "sp":
                            if (!MyShop.hasStock()) {
                                System.out.println("No stock found");
                                break;
                            }
                            do {
                                System.out.print("Enter the stock name to show all the products : ");
                                input = scan.nextLine();
                            } while ((currentStock = MyShop.selectAStock(input)) == null);

                            if (!currentStock.hasProducts()) {
                                System.out.println("This stock don't have products");
                                break;
                            }
                            System.out.println("All products of " + currentStock.getName() + " :");
                            currentStock.printProducts();
                            break;
                        case "p":
                            if (!MyShop.hasStock()) {
                                System.out.println("No stock found");
                                break;
                            }
                            do {
                                input = getUserInputString("Enter the stock name where the product is located : ");
                            } while ((currentStock = MyShop.selectAStock(input)) == null);

                            if (!currentStock.hasProducts()) {
                                System.out.println("This stock don't have products");
                                break;
                            }

                            String productInfo;
                            do {
                                input = getUserInputString("Enter a name :");
                            } while ((productInfo = currentStock.getProductInfo(input)) == null);

                            System.out.println("Product info :");
                            System.out.println(productInfo);
                            break;
                    }
                    break;
                case "e":
                    if (!MyShop.hasStock()) {
                        System.out.println("No stock found");
                        break;
                    }
                    do {
                        System.out.print("Enter the stock name where is the product you want to edit : ");
                        input = scan.nextLine();
                    } while ((currentStock = MyShop.selectAStock(input)) == null);

                    if (!currentStock.hasProducts()) {
                        System.out.println("This stock don't have products");
                        break;
                    }

                    System.out.println("Choose (a)dd or (r)emove");
                    input = scan.nextLine();
                    switch (input) {
                        case "a":
                            try {
                                String productName = getUserInputString("Enter a name : ");
                                if (currentStock.addProductQuantity(productName, getUserInputInt("Enter a quantity to add : ")))
                                    System.out.println(productName + " quantity changed with success");
                                else System.out.println(productName + " quantity not changed");
                            } catch (Exception e) {
                                System.out.println("Product not found");
                            }
                            break;
                        case "r":
                            try {
                                String productName = getUserInputString("Enter a name : ");
                                if (currentStock.subProductQuantity(productName, getUserInputInt("Enter a quantity to soustract : ")))
                                    System.out.println(productName + " quantity changed with success");
                                else
                                    System.out.println(productName + " quantity not changed (you remove to many products)");
                            } catch (Exception e) {
                                System.out.println("Product not found");
                            }
                            break;
                    }
                    break;
                case "q":
                    System.out.println("Don't save ?");
                    input = scan.nextLine();
                    if (!input.matches("^(?:1|t(?:rue)?|y(?:es)?|ok(?:ay)?)$"))
                        MyShop.save();
                    System.exit(0);
                    break;
            }
        }
    }

    static String getUserInputString(String msg) {
        String input;
        do {
            System.out.print(msg);
            input = scan.nextLine();
        } while (input.equals(""));
        return input;
    }

    static int getUserInputInt(String msg) {
        int num = 0;
        boolean isValidValue = false;
        do {
            System.out.print(msg);
            String input = scan.nextLine();
            try {
                num = Integer.parseInt(input);
                isValidValue = true;
            } catch (NumberFormatException ignored) {
            }
        } while (!isValidValue);
        return num;
    }
}
