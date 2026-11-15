package LP3;

import java.util.Arrays;

public class fractional_knapsack {
    // Helper class to store item details
    static class Item {
        int value, weight;
        double ratio;

        Item(int value, int weight) {
            this.value = value;
            this.weight = weight;
            this.ratio = (double) value / weight;
        }
    }

    public static double getMaxValue(Item[] items, int capacity) {
        // 1. Sort items by value-to-weight ratio in descending order
        Arrays.sort(items, (a, b) -> Double.compare(b.ratio, a.ratio));

        double totalValue = 0.0;
        int remainingCapacity = capacity;

        // 2. Iterate through sorted items
        for (Item item : items) {
            if (remainingCapacity == 0) {
                break;
            }

            if (item.weight <= remainingCapacity) {
                // Take the entire item
                totalValue += item.value;
                remainingCapacity -= item.weight;
            } else {
                // Take only the fraction that fits
                totalValue += item.ratio * remainingCapacity;
                remainingCapacity = 0;
            }
        }

        return totalValue;
    }

    public static void main(String[] args) {
        Item[] items = {
            new Item(60, 10),
            new Item(100, 20),
            new Item(120, 30)
        };
        int capacity = 50;

        double maxValue = getMaxValue(items, capacity);
        System.out.println("Maximum value in knapsack = " + maxValue);
        // Output: Maximum value in knapsack = 240.0
    }
}