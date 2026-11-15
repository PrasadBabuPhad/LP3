
import java.util.*;

class Node {
    char ch;
    int freq;

    Node left, right;

    Node(char ch, int freq) {
        this.ch = ch;
        this.freq = freq;
        left = right = null;
    }
}

public class HuffmanEncoding {

    // Generate Huffman Codes
    static void printCodes(Node root, String code) {

        if (root == null)
            return;

        // Leaf node contains a character
        if (root.left == null && root.right == null) {
            System.out.println(root.ch + " : " + code);
            return;
        }

        printCodes(root.left, code + "0");
        printCodes(root.right, code + "1");
    }

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of characters: ");
        int n = sc.nextInt();

        PriorityQueue<Node> pq = new PriorityQueue<>(
            (a, b) -> Integer.compare(a.freq, b.freq)
        );

        // Input characters and frequencies
        for (int i = 0; i < n; i++) {
            System.out.print("Enter character: ");
            char ch = sc.next().charAt(0);

            System.out.print("Enter frequency: ");
            int freq = sc.nextInt();

            pq.add(new Node(ch, freq));
        }

        // Build Huffman Tree using Greedy Strategy
        while (pq.size() > 1) {

            Node left = pq.poll();
            Node right = pq.poll();

            Node parent = new Node(
                '-', left.freq + right.freq
            );

            parent.left = left;
            parent.right = right;

            pq.add(parent);
        }

        // Print Huffman Codes
        Node root = pq.poll();

        System.out.println("\nHuffman Codes:");
        if (root != null) {
            if (root.left == null && root.right == null) {
                System.out.println(root.ch + " : 0");
            } else {
                printCodes(root, "");
            }
        }

        sc.close();
    }
}
