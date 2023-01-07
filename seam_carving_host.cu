#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

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
void findSeamPathByHost_1(uint8_t * inPixels, int width, int height, uint32_t * seamPath)
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
void findSeamPathByHost_2(uint8_t * inPixels, int width, int height, uint32_t * seamPath)
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
            findSeamPathByHost_1(edgeDetectImg, curWidth, height, seamPath);
        } 
        else 
        {
			findSeamPathByHost_2(edgeDetectImg, curWidth, height, seamPath);
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
	
    seamCarvingByHost(inPixels, width, height, outPixels, scale_width, improvement);
	
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

	// TODO: Seam carving input image using host
	
	// No improvement
	uchar3 * outPixelsByHostNoImprovement = (uchar3 *)malloc(scale_width * height * sizeof(uchar3)); 
	seamCarving(inPixels, width, height, outPixelsByHostNoImprovement, scale_width);

	// Improvement version 1
	uchar3 * outPixelsByHostImprovement1 = (uchar3 *)malloc(scale_width * height * sizeof(uchar3)); 
	seamCarving(inPixels, width, height, outPixelsByHostImprovement1, scale_width, false, dim3(1, 1), 1);
	
    // Write results to files
    char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    writeGrayscalePnm(edgeDetectImg, 1, width, height, concatStr(outFileNameBase, "_edgeDetect.pnm"));
	writePnm(outPixelsByHostNoImprovement, scale_width, height, concatStr(outFileNameBase, "_host.pnm"));
	writePnm(outPixelsByHostImprovement1, scale_width, height, concatStr(outFileNameBase, "_host1.pnm"));

	// Free memories
	free(inPixels);
    free(grayScaleImg);
    free(edgeDetectImg);
	free(outPixelsByHostNoImprovement);
	free(outPixelsByHostImprovement1);
}