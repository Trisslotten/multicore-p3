#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "CImg.h"
#include "timer.hpp"

using namespace cimg_library;

using ColorType = unsigned char;


__global__ void unweaveKernel(ColorType* src, ColorType* dst, int numPixels)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = 3 * i;
	if (i < numPixels)
	{
		int stride = numPixels;
		dst[i] = src[j];
		dst[i + stride] = src[j + 1];
		dst[i + 2 * stride] = src[j + 2];
	}
}

__global__ void blurKernel(ColorType* src, ColorType* blurred, int width, int height, float* mask)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix < width && iy < height)
	{
		int stride = width * height;

		float sumR = 0.f;
		float sumG = 0.f;
		float sumB = 0.f;

		for (int y = 0; y < 5; y++)
		{
			for (int x = 0; x < 5; x++)
			{
				int cx = ix + x - 2;
				int cy = iy + y - 2;
				// clamp to edge
				cx = max(0, min(width - 1, cx));
				cy = max(0, min(height - 1, cy));
				int i = cx + cy * width;

				float currMask = mask[x + y * 5];
				sumR += currMask * src[i];
				sumG += currMask * src[i+stride];
				sumB += currMask * src[i+2*stride];
			}
		}

		int i = ix + iy * width;

		blurred[i] = sumR;
		blurred[i+stride] = sumG;
		blurred[i+2*stride] = sumB;
	}
}

int main()
{
	Timer t;

	CImg<ColorType> image("cake.ppm");
	int width = image.width();
	int height = image.height();

	image.permute_axes("cxyz");
	
	
	CImg<float> mask5(5, 5);
	mask5(0, 0) = mask5(0, 4) = mask5(4, 0) = mask5(4, 4) = 1.f / 256.f;
	mask5(0, 1) = mask5(0, 3) = mask5(1, 0) = mask5(1, 4) = mask5(3, 0) = mask5(3, 4) = mask5(4, 1) = mask5(4, 3) = 4.f / 256.f;
	mask5(0, 2) = mask5(2, 0) = mask5(2, 4) = mask5(4, 2) = 6.f / 256.f;
	mask5(1, 1) = mask5(1, 3) = mask5(3, 1) = mask5(3, 3) = 16.f / 256.f;
	mask5(1, 2) = mask5(2, 1) = mask5(2, 3) = mask5(3, 2) = 24.f / 256.f;
	mask5(2, 2) = 36.f / 256.f;


	size_t imgMemSize = sizeof(ColorType)*image.size();
	size_t maskMemSize = sizeof(float) * 5 * 5;

	ColorType* d_interleavedImg = nullptr;
	ColorType* d_img = nullptr;
	ColorType* d_blurImg = nullptr;
	float* d_mask = nullptr;
	cudaMalloc((void**)&d_interleavedImg, imgMemSize);
	cudaMalloc((void**)&d_img, imgMemSize);
	cudaMalloc((void**)&d_blurImg, imgMemSize);
	cudaMalloc((void**)&d_mask, maskMemSize);
	cudaMemcpy(d_interleavedImg, image.data(), imgMemSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask, mask5.data(), maskMemSize, cudaMemcpyHostToDevice);


	int size = image.size() / 3;
	int unweaveBlockSize = 128;
	int unweaveNumBlocks = std::ceil(double(size) / unweaveBlockSize);
	unweaveKernel<<<unweaveNumBlocks, unweaveBlockSize >>>(d_interleavedImg, d_img, size);

	dim3 blurBlockSize(17, 17);
	dim3 blurNumBlocks(std::ceil(double(width) / blurBlockSize.x),
					   std::ceil(double(height) / blurBlockSize.y));

	image.permute_axes("yzcx");



	cudaDeviceSynchronize();
	t.restart();
	blurKernel<<<blurNumBlocks, blurBlockSize>>>(d_img, d_blurImg, width, height, d_mask);
	cudaDeviceSynchronize();
	double elapsed = t.elapsed();
	std::cout << "elapsed: " << elapsed << "\n";

	cudaMemcpy(image.data(), d_blurImg, imgMemSize, cudaMemcpyDeviceToHost);

	for (int i = 0; i < 3; i++)
	{
		std::cout << (int)image.data()[i] << " " << (int)image.data()[i+width*height] << " " << (int)image.data()[i + 2*width * height] << "\n";
	}

	CImgDisplay display(image, "bogdan");
	while (true)
		display.wait();

	return 0;
}