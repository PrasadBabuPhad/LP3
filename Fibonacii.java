import java.util.Scanner;

public class Fibonacii {

    // Recursive method
    static int fibonacciRecursive(int n) {
        if (n == 0)
            return 0;

        if (n == 1)
            return 1;

        return fibonacciRecursive(n - 1)
             + fibonacciRecursive(n - 2);
    }

    // Non-recursive (Iterative) method
    static void fibonacciIterative(int n) {
        int a = 0, b = 1;

        System.out.println("Non-Recursive Fibonacci Series:");

        for (int i = 0; i < n; i++) {
            System.out.print(a + " ");

            int c = a + b;
            a = b;
            b = c;
        }

        System.out.println();
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of terms: ");
        int n = sc.nextInt();

        if (n < 0) {
            System.out.println("Enter a non-negative number.");
        } else {
            // Non-recursive approach
            fibonacciIterative(n);

            // Recursive approach
            System.out.println("Recursive Fibonacci Series:");

            for (int i = 0; i < n; i++) {
                System.out.print(fibonacciRecursive(i) + " ");
            }

            System.out.println();
        }

        sc.close();
    }
}
