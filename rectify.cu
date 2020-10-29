#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gputimer.h"
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

//Parallelizatiohn scheme each thread takes task_len number of tasks, e.g. thread_0 takes first task_len, thread_1 takes the next task_len
__global__ void process(unsigned width, unsigned height, unsigned char * new_image, unsigned char * image, int thread_number) {
    unsigned char value;
   
    int thread_id = threadIdx.x;
    int task_len = width * height * 4 / thread_number;
    for (int i = thread_id * task_len; i < thread_id * task_len + task_len; i++) {
        //Rectification
        if (image[i] < 127) {
            new_image[i] = (unsigned char) 127;
        } else {
            new_image[i] = image[i];
        }
     }
    
}


int main(int argc, char* argv[]) {
    char* input_filename = argv[1];
    char* output_filename = argv[2];
    int thread_number = atoi(argv[3]);

    unsigned error;
    unsigned char* image, * new_image, * cuda_image, * cuda_new;
    unsigned width, height;

    GpuTimer timer = GpuTimer();
    
    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
 
    size_t size = width * height * 4 * sizeof(unsigned char);

    new_image = (unsigned char*)malloc(size);

    cudaMalloc((void**)&cuda_image, size);
    cudaMalloc((void**)&cuda_new, size);

    cudaMemcpy(cuda_image, image, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_new, new_image, size, cudaMemcpyHostToDevice);

    
    timer.Start();
    process<<<1,thread_number>>> (width, height, cuda_new, cuda_image, thread_number);
    cudaDeviceSynchronize();
    timer.Stop();
    printf("Num Threads %d:Time: %f", thread_number, timer.Elapsed());
   
    cudaMemcpy(new_image, cuda_new, size, cudaMemcpyDeviceToHost);
   
    lodepng_encode32_file(output_filename, new_image, width, height);
    
    free(image);
    free(new_image);
    return 0;
}