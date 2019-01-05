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

__global__ void blurKernel(ColorType* src, ColorType* blurred, int width, int height, float* mask, bool vertPass)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix < width && iy < height)
	{
		int stride = width * height;

		float sumR = 0.f;
		float sumG = 0.f;
		float sumB = 0.f;

		for (int m = 0; m < 5; m++)
		{
			int cx = ix;
			int cy = iy;
			if (vertPass)
			{
				// clamp to edge
				cy += m - 2;
				cy = max(0, min(width - 1, cy));
			}
			else
			{
				cx += m - 2;
				cx = max(0, min(width - 1, cx));
			}
			
			int i = cx + cy * width;

			float currMask = mask[m];
			sumR += currMask * src[i];
			sumG += currMask * src[i+stride];
			sumB += currMask * src[i+2*stride];
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
	
	const char* file = "cake.ppm";
	CImg<ColorType> image(file);
	int width = image.width();
	int height = image.height();

	image.permute_axes("cxyz");
	
	/*
	CImg<float> mask5(5, 5);
	mask5(0, 0) = mask5(0, 4) = mask5(4, 0) = mask5(4, 4) = 1.f / 256.f;
	mask5(0, 1) = mask5(0, 3) = mask5(1, 0) = mask5(1, 4) = mask5(3, 0) = mask5(3, 4) = mask5(4, 1) = mask5(4, 3) = 4.f / 256.f;
	mask5(0, 2) = mask5(2, 0) = mask5(2, 4) = mask5(4, 2) = 6.f / 256.f;
	mask5(1, 1) = mask5(1, 3) = mask5(3, 1) = mask5(3, 3) = 16.f / 256.f;
	mask5(1, 2) = mask5(2, 1) = mask5(2, 3) = mask5(3, 2) = 24.f / 256.f;
	mask5(2, 2) = 36.f / 256.f;
	*/

	float mask[5];
	mask[0] = mask[4] = 1.f / 16.f;
	mask[1] = mask[3] = 4.f / 16.f;
	mask[2] = 6.f / 16.f;


	size_t imgMemSize = sizeof(ColorType)*image.size();
	size_t maskMemSize = sizeof(float) * 5;

	ColorType* d_interleavedImg = nullptr;
	ColorType* d_img = nullptr;
	ColorType* d_blurImg = nullptr;
	float* d_mask = nullptr;
	cudaMalloc((void**)&d_interleavedImg, imgMemSize);
	cudaMalloc((void**)&d_img, imgMemSize);
	cudaMalloc((void**)&d_blurImg, imgMemSize);
	cudaMalloc((void**)&d_mask, maskMemSize);
	cudaMemcpy(d_interleavedImg, image.data(), imgMemSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask, mask, maskMemSize, cudaMemcpyHostToDevice);


	int size = image.size() / 3;
	int unweaveBlockSize = 128;
	int unweaveNumBlocks = std::ceil(double(size) / unweaveBlockSize);
	unweaveKernel<<<unweaveNumBlocks, unweaveBlockSize>>>(d_interleavedImg, d_img, size);

	dim3 blurBlockSize(32, 32);
	dim3 blurNumBlocks(std::ceil(double(width) / blurBlockSize.x),
					   std::ceil(double(height) / blurBlockSize.y));

	image.permute_axes("yzcx");



	cudaDeviceSynchronize();
	t.restart();
	blurKernel<<<blurNumBlocks, blurBlockSize>>>(d_img, d_blurImg, width, height, d_mask, true);
	std::swap(d_img, d_blurImg);
	blurKernel<<<blurNumBlocks, blurBlockSize>>>(d_img, d_blurImg, width, height, d_mask, false);
	cudaDeviceSynchronize();
	double elapsed = t.elapsed();
	std::cout << "elapsed: " << elapsed << "\n";

	cudaMemcpy(image.data(), d_blurImg, imgMemSize, cudaMemcpyDeviceToHost);

	CImg<ColorType> original(file);

	CImgDisplay display1(original, "original");
	CImgDisplay display2(image, "blurred");
	while (true)
	{
		display1.wait();
		display2.wait();
	}

	return 0;
}