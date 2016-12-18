import java.util.Arrays;
import java.util.List;

public class Bsearch {
	public static void main(String[] args) {
		List<Integer> list = Arrays.asList(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024);
		Integer[] array = (Integer[])list.toArray();
		int value = 4;
		int result = new ExtendedArray(array).binarySearch(value);
		System.out.println("Position of " + value + " in " + list + " is " + result);
		
		value = 0;
		result = new ExtendedArray(array).binarySearch(value);
		System.out.println("Position of " + value + " in " + list + " is " + result);
	}
}

class ExtendedArray {
	private Integer[] data;
	
	public ExtendedArray(Integer[] source) {
		this.data = source;
	}
	
	public int binarySearch(int value){
        int h = data.length - 1;
        int l = 0;
        while (h >= l) {
                int m = l + ((h - l) / 2);
                if (data[m] > value) {
                        h = m - 1;
                } else if (data[m] < value) {
                        l = m + 1;
                } else {
                        return m;
                }
        }
        return -1;
	}
}