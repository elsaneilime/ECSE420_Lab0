#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gputimer.h"
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>


//calculates the starting index of each pooling window
__device__ unsigned getIndexOfWindow(unsigned radius, unsigned grid_number) {
    return ((radius * (grid_number / radius)) * 4 * 4 + (grid_number % radius) * 8);
}

//calculates the max value of the parsed array 
__device__ unsigned find_maximum(unsigned a[], int n) {
    int c, max = 0;

    for (c = 0; c < n; c++)
        if (a[c] > max)
            max = a[c];
    return max;
}

__device__ unsigned find_index(unsigned a[], int n) {
    int c, index = 0;

    for (c = 1; c < n; c++)
        if (a[c] > a[index])
            index = c;

    return index;
}

__device__ unsigned pooling(unsigned index, unsigned color_at_pixel, unsigned width, unsigned char* image, unsigned char* new_image, unsigned thread_id) {
    //  do [pixel[i], pixel[i+1], pixel[i+width], pixel[i+width+1]]
    //  unsigned grid_r = { image[i], image[i+ 4], image[i + width*4], image[i+(width + 1)*4] }
    //  unsigned grid_g = { image[i+1], image[i+1 + 4], image[i+1 + width * 4], image[i+1 + (width + 1) * 4] }
    //  pooling_r = find_maximum(grid_r, 4)
    //  pooling_b = find_maximum(grid_g, 4);
    // new_image[i], new_image[i+1]
    for (int i = index; i < index + 4; i++) {
        unsigned grid[4] = { (unsigned) image[i], (unsigned) image[i + 4], (unsigned) image[i + width * 4], (unsigned) image[i + (width + 1) * 4] };
        new_image[color_at_pixel+(i-index)] = find_maximum(grid,4);
        
    }
}

// process<<<1, num_threads>>> (width, windows_per_thread, num_threads, cuda_image, cuda_new);
__global__ void process(unsigned width, unsigned height, unsigned windows_per_thread, unsigned num_threads, unsigned char* image, unsigned char* new_image) {
    unsigned radius = width / 2;
    for (int i = 0; i < windows_per_thread; i++) {
        unsigned index_at_new_image = (threadIdx.x * windows_per_thread + i) * 4;
        pooling(getIndexOfWindow(radius, threadIdx.x * windows_per_thread+i), (index_at_new_image), width, image, new_image, threadIdx.x);
    }
    
    // do remaining windows
    if (threadIdx.x == num_threads-1) {
        unsigned remaining_windows = (width * height / 4) % num_threads;
        if (remaining_windows > 0) {
            for (int i = 0; i < remaining_windows; i++) {
                unsigned index_at_new_image = ((threadIdx.x+1) * windows_per_thread + i) * 4;
                pooling(getIndexOfWindow(radius, (threadIdx.x + 1) * windows_per_thread + i), (index_at_new_image), width, image, new_image, threadIdx.x);
            }
         }
    }
}



int main(int argc, char* argv[]) {
    char* input_filename = argv[1];
    char* output_filename = argv[2];
    int num_threads = atoi(argv[3]);

    unsigned error;
    unsigned char* image, * new_image, * cuda_image, * cuda_new;
    unsigned width, height, num_windows, windows_per_thread;
    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
 
    size_t original_size = width * height * 4 * sizeof(unsigned char);
    size_t compressed_size = width * height * sizeof(unsigned char);
    
    GpuTimer timer = GpuTimer();
    
    new_image = (unsigned char*)malloc(compressed_size);
    cudaMalloc((void**)&cuda_image, original_size);
    cudaMalloc((void**)&cuda_new, compressed_size);

    num_windows = width * height / 4;
    windows_per_thread = num_windows / num_threads;
  
    cudaMemcpy(cuda_image, image, original_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_new, new_image, compressed_size, cudaMemcpyHostToDevice);
     
     timer.Start();
     process<<<1, num_threads>>> (width, height, windows_per_thread, num_threads, cuda_image, cuda_new);
     cudaDeviceSynchronize();
     timer.Stop();
     printf("Num Threads %d:Time: %f", num_threads, timer.Elapsed());
     
     cudaMemcpy(new_image, cuda_new, compressed_size, cudaMemcpyDeviceToHost);
   
     lodepng_encode32_file(output_filename, new_image, width/2, height/2);
    
    free(image);
    free(new_image);
    return 0;
}



// Sequential version
//int color_at_pixel = 0;
//for (int i = 0; i < height / 2; i++) {
//    unsigned* mini_grid = (unsigned*)malloc(16 * sizeof(unsigned));
//    for (int j = 0; j < width / 2; j++) {
//        int current_index = i * width * 4 * 2 + 8 * j;
//
//        if (i < 10 && j < 10)
//            printf("\nGrid(%d,%d): ", i, j);
//
//
//        for (int idx = 0; idx < 16; idx++) {
//            mini_grid[idx] = (unsigned)image[current_index + idx];  // [1,9,4,2,4,5,2,3,9,3,4,1,1,4,5,6]
//
//            if (i < 10 && j < 10)
//                printf("%d,", mini_grid[idx]);
//
//        }
//
//
//        for (int k = 0; k < 4; k++) {
//            unsigned color[4] = { mini_grid[k], mini_grid[k + 4], mini_grid[k + 8], mini_grid[k + 12] };
//            new_image[color_at_pixel] = (char)find_maximum(color, 4);
//            color_at_pixel++;
//
//            // print color array
//            if (i < 10 && j < 10) {
//                printf("\n color array: [ ");
//                for (int loop = 0; loop < 4; loop++)
//                    printf("%d,", color[loop]);
//                printf(" ]\n");
//            }
//
//        }
//    }
//}