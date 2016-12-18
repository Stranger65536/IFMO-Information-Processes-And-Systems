#include <stdio.h>
#include <stdlib.h>

int cmpint(int *pa, int pb)
{
	int a = *pa;
	int b = pb;
	if (a < b) return -1;
	if (a == b) return 0;
	if (a > b) return 1;
}

int bsearch(int Array[], int n, int key,int(*cmp)(int *, int pb))
{
	unsigned left = 1, right = n; 
	int NotFound = -1;
	if (!(Array && n > 0 && key && cmp))
		return NotFound; 

	while (left < right)
	{
		unsigned m = (left + right) / 2;

		if (cmp(Array + m, key) < 0)
			left = m + 1;      
		else
			right = m;        
	}
	return (cmp(Array + right, key) == 0) ? right : NotFound;
}

int main(void)
{
	int a[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
	int n = sizeof(a) / sizeof(*a) - 1;

	int key = 4;

	int b = bsearch(a, n, key,cmpint);
	printf("Position of %d in [ ", key);
	for (int i = 0; i <= n; i++) 
	{
		printf("%d ", a[i]);
	}
	printf("] is %d\n", b);
	
	key = 0;

	b = bsearch(a, n, key, cmpint);
	printf("Position of %d in [ ", key);
	for (int i = 0; i <= n; i++) 
	{
		printf("%d ", a[i]);
	}
	printf("] is %d\n", b);
	return 0;
}