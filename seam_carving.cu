#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

// ---------------------------------------- Global variables ----------------------------------------

// CMEM for filter
#define FILTER_WIDTH 3
__constant__ float dc_filterX[FILTER_WIDTH * FILTER_WIDTH];
__constant__ float dc_filterY[FILTER_WIDTH * FILTER_WIDTH];

// ---------------------------------------- Utility function ----------------------------------------

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

void writePnm(uchar3 * pixels, int width, int height, char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	
	fclose(f);
}

void writeGrayscalePnm(uint8_t * pixels, int numChannels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P3\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height * numChannels; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
}

float computeError(uchar3 * a1, uchar3 * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
	{
		err += abs((int)a1[i].x - (int)a2[i].x);
		err += abs((int)a1[i].y - (int)a2[i].y);
		err += abs((int)a1[i].z - (int)a2[i].z);
	}
	err /= (n * 3);
	return err;
}

void printError(uchar3 * deviceResult, uchar3 * hostResult, int width, int height)
{
	float err = computeError(deviceResult, hostResult, width * height);
	printf("Error: %f\n", err);
}

char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void printDeviceInfo()
{
	cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);

    printf("****************************\n");
}

// ---------------------------------------- Sequential code -----------------------------------------

// Convert input image into grayscale image
// uchar3 * inPixels: input image
// int width: input image width
// int height: input image height
// uint8_t * outPixels: grayscale image
void convertToGrayscaleByHost(uchar3 * inPixels, int width, int height, uint8_t * outPixels)
{
    // gray = 0.299 * red + 0.587 * green + 0.114 * blue  
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int i = r * width + c;
            outPixels[i] = 0.299f * inPixels[i].x + 0.587f * inPixels[i].y + 0.114f * inPixels[i].z;
        }
    }
}

// Convert input image into energy matrix using Edge detection with Sobel Kernel
// uint8_t * inPixels: input image after convert into grayscale
// int width: input image width
// int height: input image height
// uint8_t * energyMatrix: energy matrix
void edgeDetectionByHost(uint8_t * inPixels, int width, int height, uint8_t * energyMatrix)
{
	// X axis edge dectect
	int filterX[9] = {-1, 0, 1,
					  -2, 0, 2,
					  -1, 0, 1};
	// Y axis edge dectect
	int filterY[9] = {1, 2, 1,
					  0, 0, 0,
					 -1, -2, -1};
	int filterWidth = 3;

	for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
	{
		for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
		{
			float outPixelX = 0;
			float outPixelY = 0;
			for (int filterR = 0; filterR < filterWidth; filterR++)
			{
				for (int filterC = 0; filterC < filterWidth; filterC++)
				{
					float filterValX = filterX[filterR*filterWidth + filterC];
					float filterValY = filterY[filterR*filterWidth + filterC];

					int inPixelsR = outPixelsR - filterWidth/2 + filterR;
					int inPixelsC = outPixelsC - filterWidth/2 + filterC;
					inPixelsR = min(max(0, inPixelsR), height - 1);
					inPixelsC = min(max(0, inPixelsC), width - 1);
					uint8_t inPixel = inPixels[inPixelsR*width + inPixelsC];

					outPixelX += inPixel * filterValX;
					outPixelY += inPixel * filterValY;
				}
			}
			energyMatrix[outPixelsR*width + outPixelsC] = abs(outPixelX) + abs(outPixelY); 
		}
	}
}

// Find seam path with simple solution -> pixel value + min of 3 neightbor pixels
// uint8_t * inPixels: input after edge detect
// int width: input image width
// int height: input image height
// uint32_t * seamPath: Least significant seam
void findSeamPathByHost1(uint8_t * inPixels, int width, int height, uint32_t * seamPath)
{
    uint32_t * path;
    path = (uint32_t *)malloc((height + 1) * sizeof(uint32_t));
    memset(path, 0, (height + 1) * sizeof(uint32_t));
	uint32_t minSum = 99999;

    for (int c = 0; c < width; c++) 
    {
        path[0] = c;
        path[height] = inPixels[c];
        int idx = 0;

        for (int r = 1; r < height; r++)
        {
            if (c == 0)
            {
                int mid = r * width + c;
                int right = r * width + (c + 1);
                
                idx = mid;
                if (inPixels[right] < inPixels[idx]) idx = right;
            }
            else if (c == width - 1)
            {
                int left = r * width + (c - 1);
                int mid = r * width + c;

                idx = left;
                if (inPixels[mid] < inPixels[idx]) idx = mid;             
            }
            else 
            {
                int left = r * width + (c - 1);
                int mid = r * width + c;
                int right = r * width + (c + 1);

                idx = left;
                if (inPixels[mid] < inPixels[idx]) idx = mid;
                if (inPixels[right] < inPixels[idx]) idx = right;
            }

            path[r] = idx;
            path[height] += inPixels[idx];
        }

        if (path[height] < minSum)
        {
            memcpy(seamPath, path, (height + 1) * sizeof(uint32_t));
			minSum = path[height];
        }
    }
	
	free(path);
}

// Find seam path with dynamic programing: calculate minimum energy maxtrix then 
//                                         backtrack to find seam path
// uint8_t * inPixels: input after edge detect
// int width: input image width
// int height: input image height
// uint32_t * seamPath: Least significant seam
void findSeamPathByHost2(uint8_t * inPixels, int width, int height, uint32_t * seamPath)
{
	uint32_t * minimumEnergy, * backtrack, * tmp;
	backtrack = (uint32_t *)malloc(width * height * sizeof(uint32_t));
	tmp = (uint32_t *)malloc(height * sizeof(uint32_t));
	minimumEnergy = (uint32_t *)malloc(width * height * sizeof(uint32_t));
	
	// Top row 
	for (int c = 0; c < width; c++)
	{
		minimumEnergy[c] = inPixels[c];
	}

    // Compute minimum energy matrix
    for (int r = 1; r < height; r++) 
    {
        for (int c = 0; c < width; c++)
        {
			int idx = 0;
            if (c == 0)
            {
                int mid = (r - 1) * width + c;
                int right = (r - 1) * width + (c + 1);
                
                idx = mid;
                if (minimumEnergy[right] < minimumEnergy[idx]) idx = right;
            }
            else if (c == width - 1)
            {
                int left = (r - 1) * width + (c - 1);
                int mid = (r - 1) * width + c;

                idx = left;
                if (minimumEnergy[mid] < minimumEnergy[idx]) idx = mid;             
            }
            else 
            {
                int left = (r - 1) * width + (c - 1);
                int mid = (r - 1) * width + c;
                int right = (r - 1) * width + (c + 1);

                idx = left;
                if (minimumEnergy[mid] < minimumEnergy[idx]) idx = mid;
                if (minimumEnergy[right] < minimumEnergy[idx]) idx = right;
            }
			
			int curIdx = r * width + c;
            minimumEnergy[curIdx] = inPixels[curIdx] + minimumEnergy[idx];
			backtrack[curIdx] = idx;
        }
    }

	// Find min at bottom
	uint32_t min = minimumEnergy[(height - 1) * width];
	uint32_t minIdx = 0;
	for (int c = 1; c < width; c++) 
	{
		if (minimumEnergy[(height - 1) * width + c] < min) 
		{
			min = minimumEnergy[(height - 1) * width + c];
			minIdx = (height - 1) * width + c;
		}
	}

	// Backtrack from bottom
	seamPath[0] = minIdx;
	int curIdx = minIdx;
	for (int r = 1; r < height; r++)
	{
		seamPath[r] = backtrack[curIdx];
		curIdx = backtrack[curIdx];
	}

	// Reverse seamPath
	memcpy(tmp, seamPath, height * sizeof(uint32_t));
	int idx = 0;
	for (int i = height - 1; i >= 0; i--)
	{
		seamPath[idx] = tmp[i];
		idx++;
	}
	
	free(minimumEnergy);
	free(backtrack);
    free(tmp);
}

// Seam carving using host
// uchar3 * inPixels: input image
// int width: input image width
// int height: input image height
// int scale_width: image width after seam carving
// uchar3 * outPixels: image after seam carving
// int improvement: improvement version 0 & 1
void seamCarvingByHost(uchar3 * inPixels, int width, int height, uchar3 * outPixels, 
        int scale_width, int improvement= 0)
{
    uchar3 * img = (uchar3 *)malloc(width * height * sizeof(uchar3));
    memcpy(img, inPixels, (width * height * sizeof(uchar3)));

    if (improvement == 0)
    {
        // TODO: Host -> Find Seam path using Greedy Algorithm
        printf("\nHost");
    }
    else
    {
        // TODO: Improvement version 1 -> Find Seam path using Dynamic Programming
        printf("\nHost improvement version 1");
    }

	for (int i = 0; i < width - scale_width; i++)
    {
        int curWidth = width - i;
        uint8_t * grayScaleImg = (uint8_t *)malloc(curWidth * height * sizeof(uint8_t));
        uint8_t * edgeDetectImg = (uint8_t *)malloc(curWidth * height * sizeof(uint8_t));

		// TODO: Convert input image into grayscale image
        convertToGrayscaleByHost(img, curWidth, height, grayScaleImg);
		
        // TODO: Edge Detection
        edgeDetectionByHost(grayScaleImg, curWidth, height, edgeDetectImg);
        
        // TODO: Find Seam path and remove Seam path
        uint32_t * seamPath;
        uchar3 * temp;
        seamPath = (uint32_t *)malloc(height * sizeof(uint32_t));
        memset(seamPath, 0, height * sizeof(uint32_t));

        if (improvement == 0)
        {
            findSeamPathByHost1(edgeDetectImg, curWidth, height, seamPath);
        } 
        else 
        {
			findSeamPathByHost2(edgeDetectImg, curWidth, height, seamPath);
        }
		
		temp = (uchar3 *)malloc((curWidth - 1) * height * sizeof(uchar3));

        int idx = 0;
        for (int r = 0; r < height; r++) 
        {
            for (int c = 0; c < curWidth; c++) 
            {
                int i = r * curWidth + c;
                if (i != seamPath[r])
                {
                    temp[idx] = img[i];
                    idx++;
                }
            }
        }

        img = (uchar3 *)realloc(img, (curWidth - 1) * height * sizeof(uchar3));
        memcpy(img, temp, (curWidth - 1) * height * sizeof(uchar3));
		
		free(grayScaleImg);
		free(edgeDetectImg);
        free(seamPath);
        free(temp);
    }

    memcpy(outPixels, img, scale_width * height * sizeof(uchar3));

    free(img);
}

// ----------------------------------------- Parallel code ------------------------------------------

// Convert input image into grayscale image kernel
// uchar3 * inPixels: input image
// int width: input image width
// int height: input image height
// uint8_t * outPixels: grayscale image
__global__ void convertToGrayscaleKernel(uchar3 * inPixels, int width, int height, 
		uint8_t * outPixels)
{
    // Reminder: gray = 0.299 * red + 0.587 * green + 0.114 * blue
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;	
	
    if (r < height && c < width)
    { 
        int i = r * width + c;
        outPixels[i] = 0.299f * inPixels[i].x + 0.587f * inPixels[i].y + 0.114f * inPixels[i].z;
    }	
}

// Convert input image into grayscale image
// uchar3 * inPixels: input image
// int width: input image width
// int height: input image height
// uint8_t * outPixels: grayscale image
void convertToGrayscaleByDevice(uchar3 * inPixels, int width, int height, uint8_t * outPixels, 
		dim3 blockSize=dim3(1))
{
	// TODO: Allocate device memories
	uchar3 * d_in;
	uint8_t * d_out;
	CHECK(cudaMalloc(&d_in, width * height * sizeof(uchar3)));
    CHECK(cudaMalloc(&d_out, width * height * sizeof(uint8_t)));

	// TODO: Copy data to device memories
	CHECK(cudaMemcpy(d_in, inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));

	// TODO: Set grid size and call kernel (remember to check kernel error)
	dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
	convertToGrayscaleKernel<<<gridSize, blockSize>>>(d_in, width, height, d_out);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("ERROR: %s\n", cudaGetErrorString(err));

	// TODO: Copy result from device memories
	CHECK(cudaMemcpy(outPixels, d_out, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	// TODO: Free device memories
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
}

// Kernel convert input image into energy matrix using Edge detection with Sobel Kernel
// uint8_t * inPixels: input image after convert into grayscale
// int width: input image width
// int height: input image height
// float * filterX: x-Sobel filter
// float * filterY: y-Sobel filter
// int filterWidth: filter width
// uint8_t * energyMatrix: energy matrix
__global__ void edgeDetectionKernel1(uint8_t * inPixels, int width, int height, float * filterX,
        float * filterY, int filterWidth, uint8_t * energyMatrix)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < height && c < width) 
    {
        float outPixelX = 0;
        float outPixelY = 0;
        for (int filterR = 0; filterR < filterWidth; filterR++)
        {
            for (int filterC = 0; filterC < filterWidth; filterC++)
            {
                float filterValX = filterX[filterR*filterWidth + filterC];
                float filterValY = filterY[filterR*filterWidth + filterC];

                int inPixelsR = r - filterWidth/2 + filterR;
                int inPixelsC = c - filterWidth/2 + filterC;
                inPixelsR = min(max(0, inPixelsR), height - 1);
                inPixelsC = min(max(0, inPixelsC), width - 1);
                uint8_t inPixel = inPixels[inPixelsR*width + inPixelsC];

                outPixelX += inPixel * filterValX;
                outPixelY += inPixel * filterValY;
                }
        }
        energyMatrix[r*width + c] = abs(outPixelX) + abs(outPixelY); 
    }
}

// Kernel convert input image into energy matrix using Edge detection with Sobel Kernel
// Using CMEM for filter
// uint8_t * inPixels: input image after convert into grayscale
// int width: input image width
// int height: input image height
// uint8_t * energyMatrix: energy matrix
__global__ void edgeDetectionKernel2(uint8_t * inPixels, int width, int height, uint8_t * energyMatrix)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < height && c < width) 
    {
        float outPixelX = 0;
        float outPixelY = 0;
        for (int filterR = 0; filterR < FILTER_WIDTH; filterR++)
        {
            for (int filterC = 0; filterC < FILTER_WIDTH; filterC++)
            {
                float filterValX = dc_filterX[filterR*FILTER_WIDTH + filterC];
                float filterValY = dc_filterY[filterR*FILTER_WIDTH + filterC];

                int inPixelsR = r - FILTER_WIDTH/2 + filterR;
                int inPixelsC = c - FILTER_WIDTH/2 + filterC;
                inPixelsR = min(max(0, inPixelsR), height - 1);
                inPixelsC = min(max(0, inPixelsC), width - 1);
                uint8_t inPixel = inPixels[inPixelsR*width + inPixelsC];

                outPixelX += inPixel * filterValX;
                outPixelY += inPixel * filterValY;
                }
        }
        energyMatrix[r*width + c] = abs(outPixelX) + abs(outPixelY); 
    }
}

// Using smem when multiply matrix in Edge Detection
__global__ void edgeDetectionKernelWithSmem(uint8_t * inPixels, int width, int height, uint8_t * energyMatrix)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ uint8_t s_inPixels[];
	int share_width = blockDim.x + FILTER_WIDTH - 1;
	float share_length = (blockDim.x + FILTER_WIDTH - 1) * (blockDim.y + FILTER_WIDTH - 1);

	int nPixelsPerThread = ceil(share_length / (blockDim.x * blockDim.y));
	int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;

	int firstR = blockIdx.y * blockDim.y - FILTER_WIDTH / 2;
	int firstC = blockIdx.x * blockDim.x - FILTER_WIDTH / 2;

	for (int i = 0; i < nPixelsPerThread; i++)
	{
		int pos = threadIndex * nPixelsPerThread + i;
		if (pos >= share_length) break;

		int inPixelR = pos / share_width + firstR;
		int inPixelC = pos % share_width + firstC;
		inPixelR = min(max(0, inPixelR), height - 1);
		inPixelC = min(max(0, inPixelC), width - 1);

		s_inPixels[pos] = inPixels[inPixelR * width + inPixelC];
	}
	__syncthreads();

    if (r < height && c < width)
	{
		float outPixelX = 0;
        float outPixelY = 0;
		for (int filterR = 0; filterR < FILTER_WIDTH; filterR++)
		{
			for (int filterC = 0; filterC < FILTER_WIDTH; filterC++)
			{
				float filterValX = dc_filterX[filterR*FILTER_WIDTH + filterC];
                float filterValY = dc_filterY[filterR*FILTER_WIDTH + filterC];

				int inPixelR = threadIdx.y + filterR;
				int inPixelC = threadIdx.x + filterC;

				uint8_t inPixel = s_inPixels[inPixelR * share_width + inPixelC];

				outPixelX += inPixel * filterValX;
                outPixelY += inPixel * filterValY;
			}
		}
		energyMatrix[r * width + c] = abs(outPixelX) + abs(outPixelY);
	}
}

// Convert input image into energy matrix using Edge detection with Sobel Kernel
// Using CMEM for filter
// uint8_t * inPixels: input image after convert into grayscale
// int width: input image width
// int height: input image height
// uint8_t * energyMatrix: energy matrix
// int improvement: improvement version -> version 4 using CMEM for filter
// dim3 blockSize: Block size
void edgeDetectionByDevice(uint8_t * inPixels, int width, int height, uint8_t * energyMatrix, 
		int improvement= 2, dim3 blockSize=dim3(1))
{
    // X axis edge detect
    float filterX[9] = {-1, 0, 1,
					   -2, 0, 2,
					   -1, 0, 1};

	// Y axis edge dectect
	float filterY[9] = {1, 2, 1,
					   0, 0, 0,
					  -1, -2, -1};

	int filterWidth = 3;

	// Allocate device memories
	uint8_t * d_in, * d_energyMatrix;
    float * d_filterX, * d_filterY;
	CHECK(cudaMalloc(&d_in, width * height * sizeof(uint8_t)));
    CHECK(cudaMalloc(&d_energyMatrix, width * height * sizeof(uint8_t)));

    // Copy data to device memories
    CHECK(cudaMemcpy(d_in, inPixels, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Set grid size and call kernel
	dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

    if (improvement == 2 || improvement == 3)
    {
        // Allocate device memories
        CHECK(cudaMalloc(&d_filterX, filterWidth * filterWidth * sizeof(float)));
        CHECK(cudaMalloc(&d_filterY, filterWidth * filterWidth * sizeof(float)));

        // Copy data to device memories
        CHECK(cudaMemcpy(d_filterX, filterX, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_filterY, filterY, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));

        edgeDetectionKernel1<<<gridSize, blockSize>>>(d_in, width, height, d_filterX, d_filterY, 
                filterWidth, d_energyMatrix);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) printf("ERROR: %s\n", cudaGetErrorString(err));
    }
    else if (improvement == 5) // using SMEM in edge detection
    {
        CHECK(cudaMemcpyToSymbol(dc_filterX, filterX, filterWidth * filterWidth * sizeof(float)));
        CHECK(cudaMemcpyToSymbol(dc_filterY, filterY, filterWidth * filterWidth * sizeof(float)));

        size_t share_size = (blockSize.x + filterWidth - 1) * (blockSize.y + filterWidth - 1) * sizeof(uint8_t);
        // Call Kernel
        edgeDetectionKernelWithSmem<<<gridSize, blockSize, share_size>>>(d_in, width, height, d_energyMatrix);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) printf("ERROR: %s\n", cudaGetErrorString(err));
    }
    {
        // Copy data to device memories
        CHECK(cudaMemcpyToSymbol(dc_filterX, filterX, filterWidth * filterWidth * sizeof(float)));
        CHECK(cudaMemcpyToSymbol(dc_filterY, filterY, filterWidth * filterWidth * sizeof(float)));

        // Call Kernel
        edgeDetectionKernel2<<<gridSize, blockSize>>>(d_in, width, height, d_energyMatrix);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) printf("ERROR: %s\n", cudaGetErrorString(err));
    }

	// Copy result from device memories
	CHECK(cudaMemcpy(energyMatrix, d_energyMatrix, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    // Free device memories
    if (improvement == 2 || improvement == 3)
    {
        CHECK(cudaFree(d_filterX));
        CHECK(cudaFree(d_filterY));
    }
	
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_energyMatrix));

}

// Kernel computer minimum energy on row
// uint8_t * inPixels: energy values in current row
// int width: input image width
// int height: input image height
// int curRow: current row
// uint32_t * energyMatrix: minimum energy matrix
// uint32_t * backtrack: backtrack matrix to find seam path
__global__ void computeMinimumEnergyOnRowKernel1(uint8_t * inPixelsRow, int width, int height, 
        int curRow, uint32_t * minimumEnergyRow, uint32_t * backtrack)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c < width)
    {
        int minIdx = 0;
        
        if (c == 0)
        {
            int mid = (curRow - 1) * width + c;
            int right = (curRow - 1) * width + c + 1;
            
            minIdx = mid;
            if (minimumEnergyRow[right] < minimumEnergyRow[minIdx]) minIdx = right;
        }
        else if (c == width - 1)
        {
            int left = (curRow - 1) * width + c - 1;
            int mid = (curRow - 1) * width + c;

            minIdx = left;
            if (minimumEnergyRow[mid] < minimumEnergyRow[minIdx]) minIdx = mid; 
        }
        else 
        {
            int left = (curRow - 1) * width + c - 1;
            int mid = (curRow - 1) * width + c;
            int right = (curRow - 1) * width + c + 1;

            minIdx = left;
            if (minimumEnergyRow[mid] < minimumEnergyRow[minIdx]) minIdx = mid;
            if (minimumEnergyRow[right] < minimumEnergyRow[minIdx]) minIdx = right;
        }
		
        minimumEnergyRow[curRow * width + c] = inPixelsRow[curRow * width + c] + minimumEnergyRow[minIdx];
        backtrack[curRow * width + c] = minIdx;
    }
}

// Kernel computer minimum energy on row using SMEM
// uint8_t * inPixels: energy values in current row
// int width: input image width
// int height: input image height
// int curRow: current row
// uint32_t * energyMatrix: minimum energy matrix
// uint32_t * backtrack: backtrack matrix to find seam path
__global__ void computeMinimumEnergyOnRowKernel2(uint8_t * inPixelsRow, int width, int height, 
        int curRow, uint32_t * minimumEnergyRow, uint32_t * backtrack)
{
    extern __shared__ uint8_t s_inPixelsRow[];
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c < width)
        s_inPixelsRow[c] = inPixelsRow[curRow * width + c];
    __syncthreads();
    
    if (c < width)
    {
        int minIdx = 0;
        
        if (c == 0)
        {
            int mid = (curRow - 1) * width + c;
            int right = (curRow - 1) * width + c + 1;
            
            minIdx = mid;
            if (minimumEnergyRow[right] < minimumEnergyRow[minIdx]) minIdx = right;
        }
        else if (c == width - 1)
        {
            int left = (curRow - 1) * width + c - 1;
            int mid = (curRow - 1) * width + c;

            minIdx = left;
            if (minimumEnergyRow[mid] < minimumEnergyRow[minIdx]) minIdx = mid; 
        }
        else 
        {
            int left = (curRow - 1) * width + c - 1;
            int mid = (curRow - 1) * width + c;
            int right = (curRow - 1) * width + c + 1;

            minIdx = left;
            if (minimumEnergyRow[mid] < minimumEnergyRow[minIdx]) minIdx = mid;
            if (minimumEnergyRow[right] < minimumEnergyRow[minIdx]) minIdx = right;
        }
		
        minimumEnergyRow[curRow * width + c] = s_inPixelsRow[c] + minimumEnergyRow[minIdx];
        backtrack[curRow * width + c] = minIdx;
    }
}

// Find seam path using device 
// uint8_t * inPixels: energy values in current row
// int width: input image width
// int height: input image height
// uint32_t * seamPath: Least significant seam
// int improvement: improvement version 2 -> 4
// dim3 blockSize: Block size
void findSeamPathByDevice(uint8_t * inPixels, int width, int height, uint32_t * seamPath,
        int improvement= 2, dim3 blockSize= dim3(1))
{
	uint32_t * minimumEnergy, * backtrack, * tmp;
	backtrack = (uint32_t *)malloc(width * height * sizeof(uint32_t));
	tmp = (uint32_t *)malloc(height * sizeof(uint32_t));
	minimumEnergy = (uint32_t *)malloc(width * height * sizeof(uint32_t));

	// Top row 
	for (int c = 0; c < width; c++)
	{
		minimumEnergy[c] = inPixels[c];
	}

    uint32_t * d_backtrack;
    CHECK(cudaMalloc(&d_backtrack, width * height * sizeof(uint32_t)));

    uint32_t * d_minimumEnergy;
    uint8_t * d_in;

    CHECK(cudaMalloc(&d_minimumEnergy, width * height * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_in, width * height * sizeof(uint8_t)));

    CHECK(cudaMemcpy(d_minimumEnergy, minimumEnergy, width * height * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_in, inPixels, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

    dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

    if (improvement == 2 || improvement == 5)
    {   
        // TODO: Improverment version 2 -> Parallel
        for (int r = 1; r < height; r++)
        {
            computeMinimumEnergyOnRowKernel1<<<gridSize, blockSize>>>(d_in, width, height,
                        r, d_minimumEnergy, d_backtrack);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("ERROR: %s\n", cudaGetErrorString(err));
        }
    }
    else
    {
        // TODO: Improverment version 3 -> SMEM & Improverment version 4 -> SMEM + CMEM
        size_t smem = width * sizeof(uint8_t);
        for (int r = 1; r < height; r++)
        {
            computeMinimumEnergyOnRowKernel2<<<gridSize, blockSize, smem>>>(d_in, width, height,
                        r, d_minimumEnergy, d_backtrack);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) printf("ERROR: %s\n", cudaGetErrorString(err));
        }
    }

    CHECK(cudaMemcpy(backtrack, d_backtrack, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_backtrack));

    CHECK(cudaMemcpy(minimumEnergy, d_minimumEnergy, width * height* sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_minimumEnergy));
    CHECK(cudaFree(d_in));

	// Find min at bottom
	uint32_t min = minimumEnergy[(height - 1) * width];
	uint32_t minIdx = 0;
	for (int c = 1; c < width; c++) 
	{
		if (minimumEnergy[(height - 1) * width + c] < min) 
		{
			min = minimumEnergy[(height - 1) * width + c];
			minIdx = (height - 1) * width + c;
		}
	}

	// Backtrack from bottom
	seamPath[0] = minIdx;
    int curIdx = minIdx;
	for (int r = 1; r < height; r++)
	{
        seamPath[r] = backtrack[curIdx];
        curIdx = backtrack[curIdx];
	}

	// Reverse seamPath
	memcpy(tmp, seamPath, height * sizeof(uint32_t));
	int idx = 0;
	for (int i = height - 1; i >= 0; i--)
	{
		seamPath[idx] = tmp[i];
		idx++;
	}

	free(minimumEnergy);
	free(backtrack);
    free(tmp);
}

// Seam carving using device
// uchar3 * inPixels: input image
// int width: input image width
// int height: input image height
// int scale_width: image width after seam carving
// uchar3 * outPixels: result
// int improvement: improvement version 2 -> 4 <=> improvement = 2 -> 4
// -> Improvement version 2: Parallel code
// -> Improvement version 3: Using SMEM for storing image matrix
// -> Improvement version 4: Using both SMEM and CMEM for storing kernel filter
void seamCarvingByDevice(uchar3 * inPixels, int width, int height, uchar3 * outPixels, 
        int scale_width, int improvement= 2, dim3 blockSize=dim3(1))
{
    uchar3 * img = (uchar3 *)malloc(width * height * sizeof(uchar3));
    memcpy(img, inPixels, (width * height * sizeof(uchar3)));

    if (improvement == 2)
    {
        // TODO: Improverment version 2 -> Parallel
        printf("\nParallel improvement version 2: migrate from host to cuda implementation");
    } 
    else if (improvement == 3)
    {
        // TODO: Improvement version 3 -> SMEM
        printf("\nParallel improvement version 3: using SMEM when finding Seam Path");
    }
    else if (improvement == 5)
    {
        printf("\nParallel improvement version 5: using SMEM in Edge Detection");
    }
    else 
    {
        // TODO: Improvement version 4 -> SMEM and CMEM
        printf("\nParallel improvement version 4: Using both SMEM and CMEM when Finding Seam Path");
    }

	for (int i = 0; i < width - scale_width; i++)
    {
        int curWidth = width - i;
        uint8_t * grayScaleImg = (uint8_t *)malloc(curWidth * height * sizeof(uint8_t));
        uint8_t * edgeDetectImg = (uint8_t *)malloc(curWidth * height * sizeof(uint8_t));

		// TODO: Convert input image into grayscale image
        convertToGrayscaleByDevice(img, curWidth, height, grayScaleImg, blockSize);
		
        // TODO: Edge Detection
        edgeDetectionByDevice(grayScaleImg, curWidth, height, edgeDetectImg, improvement, blockSize);
        
        // TODO: Find Seam path and remove Seam path
        uint32_t * seamPath;
        uchar3 * temp;
        seamPath = (uint32_t *)malloc(height * sizeof(uint32_t));
        memset(seamPath, 0, height * sizeof(uint32_t));

        findSeamPathByDevice(edgeDetectImg, curWidth, height, seamPath, improvement, blockSize);
		
		temp = (uchar3 *)malloc((curWidth - 1) * height * sizeof(uchar3));

        int idx = 0;
        for (int r = 0; r < height; r++) 
        {
            for (int c = 0; c < curWidth; c++) 
            {
                int i = r * curWidth + c;
                if (i != seamPath[r])
                {
                    temp[idx] = img[i];
                    idx++;
                }
            }
        }

        img = (uchar3 *)realloc(img, (curWidth - 1) * height * sizeof(uchar3));
        memcpy(img, temp, (curWidth - 1) * height * sizeof(uchar3));
		
		free(grayScaleImg);
		free(edgeDetectImg);
        free(seamPath);
        free(temp);
    }

    memcpy(outPixels, img, scale_width * height * sizeof(uchar3));

    free(img);
}

// ----------------------------------------- Seam Carving -------------------------------------------

// Seam carving function for all case
// uchar3 * inPixels: input image
// int width: input image width
// int height: input image height
// int scale_width: image width after seam carving
// uchar3 * outPixels: result after seam carving	
// int improvement: improvement version	
void seamCarving(uchar3 * inPixels, int width, int height, uchar3 * outPixels, int scale_width, 
        bool useDevice= false, dim3 blockSize= dim3(1, 1), int improvement= 0)
{
	GpuTimer timer;
	timer.Start();
	
	if (useDevice == false)	// Use host
	{
		// TODO: Seam carving using host
        seamCarvingByHost(inPixels, width, height, outPixels, scale_width, improvement);
	}
	else // Use device
	{
		// TODO: Seam carving using device
		seamCarvingByDevice(inPixels, width, height, outPixels, scale_width, improvement, blockSize);
	}
	
	timer.Stop();
    float time = timer.Elapsed();
	printf("\nRun time: %f ms\n", time);
}

// --------------------------------------------- Main -----------------------------------------------

int main(int argc, char ** argv)
{
	if (argc != 3 && argc != 4 && argc != 6)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	printDeviceInfo();
    
	// Read input image file
	int width, height;
	uchar3 * inPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("\nInput image size (width x height): %i x %i\n", width, height);
    float scale_rate = 0.85;

    if (argc >= 4) 
    {
        scale_rate = atof(argv[3]);
    }
    int scale_width = width * scale_rate;
    printf("Output image size (width x height): %i x %i\n", scale_width, height);

    uint8_t * grayScaleImg = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    uint8_t * edgeDetectImg = (uint8_t *)malloc(width * height * sizeof(uint8_t));

	// TODO: Convert input image into grayscale image
    convertToGrayscaleByHost(inPixels, width, height, grayScaleImg);
		
    // TODO: Edge Detection
    edgeDetectionByHost(grayScaleImg, width, height, edgeDetectImg);

	// Seam carving input image using host
	
	// No improvement
	uchar3 * outPixelsByHostNoImprovement = (uchar3 *)malloc(scale_width * height * sizeof(uchar3)); 
	seamCarving(inPixels, width, height, outPixelsByHostNoImprovement, scale_width);

	// Improvement version 1
	uchar3 * outPixelsByHostImprovement1 = (uchar3 *)malloc(scale_width * height * sizeof(uchar3)); 
	seamCarving(inPixels, width, height, outPixelsByHostImprovement1, scale_width, false, dim3(1, 1), 1);
	
    // Seam carving input image using device
    dim3 blockSize(32, 32); // Default
	if (argc == 6)
	{
		blockSize.x = atoi(argv[3]);
		blockSize.y = atoi(argv[4]);
	}	
	
	// Improvement version 2
	uchar3 * outPixelsByDeviceImprovement2 = (uchar3 *)malloc(scale_width * height * sizeof(uchar3));
	seamCarving(inPixels, width, height, outPixelsByDeviceImprovement2, scale_width, true, blockSize, 2);
	
	// Improvement version 3
	uchar3 * outPixelsByDeviceImprovement3 = (uchar3 *)malloc(scale_width * height * sizeof(uchar3));
	seamCarving(inPixels, width, height, outPixelsByDeviceImprovement3, scale_width, true, blockSize, 3);
	
	// Improvement version 4
	uchar3 * outPixelsByDeviceImprovement4 = (uchar3 *)malloc(scale_width * height * sizeof(uchar3));
	seamCarving(inPixels, width, height, outPixelsByDeviceImprovement4, scale_width, true, blockSize, 4);

    // Improvement version 5
	uchar3 * outPixelsByDeviceImprovement5 = (uchar3 *)malloc(scale_width * height * sizeof(uchar3));
	seamCarving(inPixels, width, height, outPixelsByDeviceImprovement5, scale_width, true, blockSize, 5);

    // Write results to files
    char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    writeGrayscalePnm(edgeDetectImg, 1, width, height, concatStr(outFileNameBase, "_edgeDetect.pnm"));
	writePnm(outPixelsByHostNoImprovement, scale_width, height, concatStr(outFileNameBase, "_host.pnm"));
	writePnm(outPixelsByHostImprovement1, scale_width, height, concatStr(outFileNameBase, "_host1.pnm"));
	writePnm(outPixelsByDeviceImprovement2, scale_width, height, concatStr(outFileNameBase, "_device2.pnm"));
	writePnm(outPixelsByDeviceImprovement3, scale_width, height, concatStr(outFileNameBase, "_device3.pnm"));
	writePnm(outPixelsByDeviceImprovement4, scale_width, height, concatStr(outFileNameBase, "_device4.pnm"));
	writePnm(outPixelsByDeviceImprovement5, scale_width, height, concatStr(outFileNameBase, "_device5.pnm"));

	// Free memories
	free(inPixels);
    free(grayScaleImg);
    free(edgeDetectImg);
	free(outPixelsByHostNoImprovement);
	free(outPixelsByHostImprovement1);
	free(outPixelsByDeviceImprovement2);
	free(outPixelsByDeviceImprovement3);
	free(outPixelsByDeviceImprovement4);
	free(outPixelsByDeviceImprovement5);
}