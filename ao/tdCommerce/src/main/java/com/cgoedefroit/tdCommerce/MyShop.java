package com.cgoedefroit.tdCommerce;

import java.util.*;

public class MyShop {
    private static Scanner scan;
    private static final Set<Stock> stocks = new HashSet<>();

    public static void main(String[] args) {
        System.out.println("Welcome in MyShop");
        scan = new Scanner(System.in);
        Stock currentStock;
        while (true) {
            System.out.println("Choose your action (c)reate, (s)how, (q)uit");
            String input = scan.nextLine();
            switch (input) {
                case "c":
                    System.out.println("Choose what you want to create (s)tock, (p)roduct");
                    input = scan.nextLine();
                    switch (input) {
                        case "s":
                            System.out.println("Create a stock");
                            String stockName = getUserInputString("Enter a name : ");
                            String address = getUserInputString("Enter an address : ");
                            if (stocks.add(new Stock(stockName, address)))
                                System.out.println("New stock created with success");
                            else System.out.println("A stock named \"" + stockName + "\" alrady exist");
                            break;
                        case "p":
                            if (stocks.size() == 0) {
                                System.out.println("You need to create stock first");
                                break;
                            }
                            System.out.println("Create a product");
                            do {
                                System.out.print("Enter the stock name where the product will be create : ");
                                input = scan.nextLine();
                            } while ((currentStock = selectAStock(input)) == null);

                            String productName = getUserInputString("Enter a name : ");
                            int quantity = getUserInputInt("Enter a quantity : ");

                            if (currentStock.addProduct(productName, quantity))
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
                            if (stocks.size() == 0) {
                                System.out.println("No stock found");
                                break;
                            }
                            System.out.println("All stocks :");
                            stocks.forEach(stock -> System.out.println(" * " + stock.toString()));
                            break;
                        case "sp":
                            if (stocks.size() == 0) {
                                System.out.println("No stock found");
                                break;
                            }
                            do {
                                System.out.print("Enter the stock name to show all the products : ");
                                input = scan.nextLine();
                            } while ((currentStock = selectAStock(input)) == null);

                            if (!currentStock.hasProducts()) {
                                System.out.println("This stock don't have products");
                                break;
                            }
                            System.out.println("All products of " + currentStock.getName() + " :");
                            currentStock.printProducts();
                            break;
                        case "p":
                            if (stocks.size() == 0) {
                                System.out.println("No stock found");
                                break;
                            }
                            do {
                                input = getUserInputString("Enter the stock name where the product is located : ");
                            } while ((currentStock = selectAStock(input)) == null);

                            if (!currentStock.hasProducts()) {
                                System.out.println("This stock don't have products");
                                break;
                            }

                            Product product;
                            do {
                                input = getUserInputString("Enter a name :");
                            } while ((product = currentStock.getProduct(input)) == null);

                            System.out.println("Product info :");
                            System.out.println(product);
                            break;
                    }
                    break;
                case "q":
                    System.exit(0);
                    break;
            }
        }
    }

    static Stock selectAStock(String name) {
        for (Stock stock : stocks)
            if (Objects.equals(stock.getName(), name))
                return stock;
        return null;
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
