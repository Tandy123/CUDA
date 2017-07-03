#include "../common/book.h"
#define N 100000

__global__ void add(int *a, int *b, int *c) {
	int tid = blockIdx.x;
	if (tid < N) {
		c[tid] = a[tid] + b[tid];
	}
}
int main(void) {
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	//在GPU上分配内存
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	//在CPU上为数组a和b赋值
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}
	//将数组a和b复制到GPU
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyHostToDevice));

	add <<<N, 1 >>> (dev_a, dev_b, dev_c);
	
	//将数组c从GPU复制到CPU
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
