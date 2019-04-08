#include <stdio.h>
#include <stdlib.h>

int main()
{
	int matrix[5][5] = {{1,1,1,1,0},{1,1,1,0,0},{0,0,1,0,0},{1,0,1,1,1},{1,1,1,1,1}};
	int row_sum_vec[5] = {};
	int col_sum_vec[5] = {};
	int n_people = 5;
	int celeb_index = 2; // to verify

	int i = 0;
	int j = 0;

	for(i=0;i<n_people;i++)
	{
		int row_sum = 0;
		for(j=0;j<n_people;j++)
		{	
			row_sum += matrix[i][j];
			col_sum_vec[j] = col_sum_vec[j] + matrix[i][j];
		}
		row_sum_vec[i] = row_sum;
	}
	int calculated_index = 0;
	for(i=0;i<n_people;i++)
	{
		if (col_sum_vec[i] == n_people && row_sum_vec[i] == 1) calculated_index = i;
	}	 
	printf("Celebrity is located at index: %d.\n",calculated_index);
	return 0;
}

