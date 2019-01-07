#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "CImg.h"
#include "timer.hpp"

using namespace cimg_library;

using ColorType = unsigned char;

const int BLUR_BLOCK_SIDE = 16;


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


__constant__ float c_mask[5];

__global__ void blurKernel(ColorType* src, ColorType* blurred, int width, int height, bool vertPass)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	__shared__ ColorType cachedR[BLUR_BLOCK_SIDE*(BLUR_BLOCK_SIDE + 4)];
	__shared__ ColorType cachedG[BLUR_BLOCK_SIDE*(BLUR_BLOCK_SIDE + 4)];
	__shared__ ColorType cachedB[BLUR_BLOCK_SIDE*(BLUR_BLOCK_SIDE + 4)];


	if (ix < width && iy < height)
	{
		int stride = width * height;
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		if (vertPass)
		{
			int index = 2 * blockDim.x + tx + ty * blockDim.x;
			cachedR[index] = src[ix + iy * width];
			cachedG[index] = src[ix + iy * width + stride];
			cachedB[index] = src[ix + iy * width + 2*stride];

			if (ty < 2 || ty >= blockDim.y - 2)
			{
				int haloIndex = tx + ty * blockDim.x;
				int cy = iy;
				if (ty < 2)
				{
					cy -= 2;
				}
				else if (ty >= blockDim.y - 2)
				{
					haloIndex += 4 * blockDim.x;
					cy += 2;
				}
				cy = max(0, min(height - 1, cy));
				cachedR[haloIndex] = src[ix + cy * width];
				cachedG[haloIndex] = src[ix + cy * width + stride];
				cachedB[haloIndex] = src[ix + cy * width + 2 * stride];
			}
		}
		else
		{
			int index = tx + 2 + ty * (4 + blockDim.x);
			cachedR[index] = src[ix + iy * width];
			cachedG[index] = src[ix + iy * width + stride];
			cachedB[index] = src[ix + iy * width + 2 * stride];

			if (tx < 2 || tx >= blockDim.x - 2)
			{
				int haloIndex = index;
				int cx = ix;
				if (tx < 2)
				{
					haloIndex -= 2;
					cx -= 2;
				}
				else
				{
					haloIndex += 2;
					cx += 2;
				}
				cx = max(0, min(width - 1, cx));
				cachedR[haloIndex] = src[cx + iy * width];
				cachedG[haloIndex] = src[cx + iy * width + stride];
				cachedB[haloIndex] = src[cx + iy * width + 2 * stride];
			}
		}

		__syncthreads();


		float sumR = 0.f;
		float sumG = 0.f;
		float sumB = 0.f;


		for (int m = 0; m < 5; m++)
		{
			int index;
			if (vertPass)
			{
				index = 2 * blockDim.x + tx + (ty + m - 2) * blockDim.x;
			}
			else
			{
				index = tx + m + ty * (4 + blockDim.x);
			}

			float currMask = c_mask[m];
			sumR += currMask * cachedR[index];
			sumG += currMask * cachedG[index];
			sumB += currMask * cachedB[index];
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

	// CImg already stores image data as RR...GG...BB...
	// this function call changes the layout to RGBRGBRGB... 
	// but the image can't be displayed now. The later call to the same function 
	// restores the image to a displayable state.
	// source: https://www.codefull.org/2014/11/cimg-does-not-store-pixels-in-the-interleaved-format/
	image.permute_axes("cxyz");


	t.restart();

	float mask[5];
	mask[0] = mask[4] = 1.f / 16.f;
	mask[1] = mask[3] = 4.f / 16.f;
	mask[2] = 6.f / 16.f;

	size_t imgMemSize = sizeof(ColorType)*image.size();
	size_t maskMemSize = sizeof(float) * 5;

	ColorType* d_interleavedImg = nullptr;
	ColorType* d_img = nullptr;
	ColorType* d_blurImg = nullptr;
	cudaMalloc((void**)&d_interleavedImg, imgMemSize);
	cudaMalloc((void**)&d_img, imgMemSize);
	cudaMalloc((void**)&d_blurImg, imgMemSize);
	cudaMemcpy(d_interleavedImg, image.data(), imgMemSize, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_mask, mask, maskMemSize);

	int size = image.size() / 3;
	int unweaveBlockSize = 1024;
	int unweaveNumBlocks = std::ceil(double(size) / unweaveBlockSize);
	unweaveKernel<<<unweaveNumBlocks, unweaveBlockSize>>>(d_interleavedImg, d_img, size);

	image.permute_axes("yzcx");

	//cudaDeviceSynchronize();

	dim3 blurBlockSize(BLUR_BLOCK_SIDE, BLUR_BLOCK_SIDE);
	dim3 blurNumBlocks(std::ceil(double(width) / blurBlockSize.x),
					   std::ceil(double(height) / blurBlockSize.y));
	blurKernel<<<blurNumBlocks, blurBlockSize>>>(d_img, d_blurImg, width, height, true);
	std::swap(d_img, d_blurImg);
	blurKernel<<<blurNumBlocks, blurBlockSize>>>(d_img, d_blurImg, width, height, false);
	//cudaDeviceSynchronize();

	cudaMemcpy(image.data(), d_blurImg, imgMemSize, cudaMemcpyDeviceToHost);

	std::cout << "elapsed: " << t.elapsed() << "\n";
	/*
	CImg<ColorType> original(file);

	CImgDisplay display1(original, "original");
	CImgDisplay display2(image, "blurred");
	while (true)
	{
		display1.wait();
		display2.wait();
	}
	*/
	return 0;
}