#include <stdio.h>
#include <stdint.h>

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

// Convert input image into energy matrix using Edge detection
// uchar3 * inPixels: input image
// int width: input image width
// int height: input image height
// uchar3 * energyMatrix: energy matrix
void edgeDetectionByHost(uint8_t * inPixels, int width, int height, uint8_t * energyMatrix)
{
	// TODO
}

// Seam carving using host
// uchar3 * inPixels: input image
// int width: input image width
// int height: input image height
// int scale_width: image width after seam carving
// uchar3 * outPixels: image after seam carving
// int improvement: improvement version 1 if improvement = 1 
void seamCarvingByHost(uint8_t * inPixels, int width, int height, uchar3 * outPixels, 
        int scale_width, int improvement= 0)
{
	// TODO: Convert input image into grayscale image
	
    for (int i = 0; i < width - scale_width; i++)
    {
        // TODO: Edge Detection
        
        // TODO: Find Seam path
        if (improvement == 0)
        {
            // TODO: Find Seam path using Greedy Algorithm
        } 
        else 
        {
            // TODO: Improvement version 1 -> Find Seam path using Dynamic Programming
        }
        
        // TODO: Remove Seam path
    }
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
	size_t nBytes = width * height * sizeof(uint8_t);
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
	CHECK(cudaMemcpy(outPixels, d_out, nBytes, cudaMemcpyDeviceToHost));

	// TODO: Free device memories
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
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
void seamCarvingByDevice(uint8_t * inPixels, int width, int height, uint8_t * outPixels, 
        int scale_width, int improvement= 0)
{
	// TODO: Convert input image into grayscale image
	
	// TODO: Edge Detection
	
	// TODO: Find Seam path
	
	// TODO: Remove Seam path
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
	}
	else // Use device
	{
		// TODO: Seam carving using device
	}
	
	timer.Stop();
    float time = timer.Elapsed();
	printf("Run time: %f ms\n", time);
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
	printf("\nImage size (width x height): %i x %i\n", width, height);
    float scale_rate = 0.8;

    if (argc >= 4) 
    {
        scale_rate = atof(argv[3]);
    }
    int scale_width = width * scale_rate;

	// Seam carving input image using host
	
	// No improvement
	uchar3 * outPixelsByHostNoImprovement = (uchar3 *)malloc(width * height * sizeof(uchar3)); 
	seamCarving(inPixels, width, height, outPixelsByHostNoImprovement, scale_width);

	// Improvement version 1
	uchar3 * outPixelsByHostImprovement1 = (uchar3 *)malloc(width * height * sizeof(uchar3)); 
	seamCarving(inPixels, width, height, outPixelsByHostImprovement1, scale_width, false, dim3(1, 1), 1);
	printError(outPixelsByHostImprovement1, outPixelsByHostNoImprovement, width, height);
	
    // Seam carving input image using device
    dim3 blockSize(32, 32); // Default
	if (argc == 6)
	{
		blockSize.x = atoi(argv[3]);
		blockSize.y = atoi(argv[4]);
	}	
	
	// Improvement version 2
	uchar3 * outPixelsByDeviceImprovement2 = (uchar3 *)malloc(width * height * sizeof(uchar3));
	seamCarving(inPixels, width, height, outPixelsByDeviceImprovement2, scale_width, true, blockSize, 2);
	printError(outPixelsByDeviceImprovement2, outPixelsByHostNoImprovement, width, height);
	
	// Improvement version 3
	uchar3 * outPixelsByDeviceImprovement3 = (uchar3 *)malloc(width * height * sizeof(uchar3));
	seamCarving(inPixels, width, height, outPixelsByDeviceImprovement3, scale_width, true, blockSize, 3);
	printError(outPixelsByDeviceImprovement3, outPixelsByHostNoImprovement, width, height);
	
	// Improvement version 4
	uchar3 * outPixelsByDeviceImprovement4 = (uchar3 *)malloc(width * height * sizeof(uchar3));
	seamCarving(inPixels, width, height, outPixelsByDeviceImprovement4, scale_width, true, blockSize, 4);
	printError(outPixelsByDeviceImprovement4, outPixelsByHostNoImprovement, width, height);

    // Write results to files
    char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	writePnm(outPixelsByHostNoImprovement, width, height, concatStr(outFileNameBase, "_host.pnm"));
	writePnm(outPixelsByHostImprovement1, width, height, concatStr(outFileNameBase, "_host1.pnm"));
	writePnm(outPixelsByDeviceImprovement2, width, height, concatStr(outFileNameBase, "_device2.pnm"));
	writePnm(outPixelsByDeviceImprovement3, width, height, concatStr(outFileNameBase, "_device3.pnm"));
	writePnm(outPixelsByDeviceImprovement4, width, height, concatStr(outFileNameBase, "_device4.pnm"));

	// Free memories
	free(inPixels);
	free(outPixelsByHostNoImprovement);
	free(outPixelsByHostImprovement1);
	free(outPixelsByDeviceImprovement2);
	free(outPixelsByDeviceImprovement3);
	free(outPixelsByDeviceImprovement4);
}