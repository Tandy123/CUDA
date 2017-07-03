#include<iostream>
#include "../common/book.h"

__global__ void add(int a, int b, int *c) {
	*c = a + b;
}

int main()
{
	int c;
	int *dev_c;
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));
	add <<<1, 1 >>> (2, 7, dev_c);
	//printf("2 + 7 = %d\n", *dev_c);//error
	HANDLE_ERROR(cudaMemcpy(&c,
		dev_c, 
		sizeof(int),
		cudaMemcpyDeviceToHost));
	printf("2 + 7 = %d\n", c);

	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));
	printf("Device Count: %d\n", count);
	cudaDeviceProp prop;
	for (int i = 0; i < count; i++) {
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
		printf("Name: %s\n", prop.name);
	}
	int dev;
	HANDLE_ERROR(cudaGetDevice(&dev));
	printf("ID of current CUDA device: %d\n", dev);

	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 3;
	HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
	printf("ID of CUDA device closest to revision 1.3: %d\n", dev);
	HANDLE_ERROR(cudaSetDevice(dev));
	cudaFree(dev_c);
	return 0;
}