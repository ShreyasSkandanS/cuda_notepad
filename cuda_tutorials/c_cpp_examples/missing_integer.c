#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// Add -lm while compiling : gcc -o output.o missing_integer.c -lm

/* Given an array with length n-1 which contains integers of the range 1 to n. Each element is distinct and appears only once. One integer is missing. Find the missing integer in linear time using O(1) memory. Now two integers are missing, find them out in linear time using O(1) memory. How about three? */

int main()
{
	// One integer is missing
	printf("---------- ONE MISSING INTEGER ----------\n");
	int arr[10] = {1,2,3,4,5,6,8,9,10,11};
	int n_minus_1 = sizeof(arr)/sizeof(arr[0]);
	int n = n_minus_1 + 1;

	int arr_sum = 0;
	int i = 0;	
	for(i=0;i<n_minus_1;i++)
	{
		arr_sum += arr[i];
	}

	int actual_sum = n*(n+1)/2;

	int missing_value = actual_sum - arr_sum;
	printf("The single missing number is %d\n",missing_value);
	
	// Two integers are missing
	printf("---------- TWO MISSING INTEGERS ---------\n");
	float arr2[8] = {10,2,3,5,7,8,9,1};
	int n_minus_2 = sizeof(arr2)/sizeof(arr2[0]);
	int n2 = n_minus_2 + 2;
	float arr2_sum = 0;
	float arr2_prod = 1;
	for(i=0;i<n_minus_2;i++)
	{
		arr2_sum += arr2[i];
		arr2_prod *= arr2[i];
	}
	int actual_prod = 1;
	for(i=1;i<=n2;i++)
	{
		actual_prod *= i;
	}
	float actual_sum2 = n2*(n2+1)/2;
	//printf("Sum of all %d integers is %.6f\n",n_minus_2,arr2_sum);
	//printf("Product of all %d integers is %.6f\n",n_minus_2,arr2_prod);
	//printf("Actual sum is %.6f and actual product is %d\n",actual_sum2,actual_prod);
	float x_plus_y = actual_sum2 - arr2_sum;
	float xy = actual_prod/arr2_prod;

	float diffsqr = sqrt(x_plus_y * x_plus_y - 4*xy);
	//printf("The difference of x and y is %.6f\n",diffsqr);
	// We found (x+y) and we found (x-y). Solve simultaneous equations to get (x) and (y)
	int a = (int)(x_plus_y + diffsqr)/2;
	int b = (int)(x_plus_y - a);
	printf("The two missing numbers are %d and %d\n",a,b);
	
	return 0;
}
	 
	 
