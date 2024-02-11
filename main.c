/*
 * Ozlem KUCUKSAGIR
 * 19050111021
 *
 * */



#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <math.h>

void sequentialPowerIteration(float *matrixA, float *vectorB, int size, int iterations, float *lambda);


void sequentialShiftedPowerIteration(float *matrixA, float *vectorB, int size, int iterations, float *eigenvalue);

void
matrixVectorMultiply(cl_context context, cl_device_id device, cl_command_queue queue, cl_mem bufferA, cl_mem bufferB,
                     cl_mem bufferResult, int rows, int cols);

void
parallelNorm(cl_context context, cl_device_id device, cl_command_queue queue, cl_mem bufferVector, cl_mem bufferResult,
             int size);

void matrixSubtraction(cl_context context, cl_device_id device, cl_command_queue queue, cl_mem bufferMatrixA,
                       cl_mem bufferMatrixB, cl_mem bufferResult, int rows, int cols);

void powerMethodOpenCL(cl_context context, cl_device_id device, cl_command_queue queue, float *matrixA, float *vectorB,
                       size_t size, int iterations, float *lambda);

void shiftedPowerMethod(cl_context context, cl_device_id device, cl_command_queue queue, float *matrixA, float *vectorB,
                        cl_mem bufferVector, size_t size, int iterations, float *lambda);

void cleanupOpenCL(cl_context context, cl_command_queue queue, cl_mem bufferA, cl_mem bufferB, cl_mem bufferResult,
                   cl_kernel kernel, cl_program program);


const char *matrixVectorSubtractionKernelSource =
        "__kernel void matrix_vector_subtraction(__global const float *matrix, __global const float *vector, __global float *result, int size) {\n"
        "    int gid = get_global_id(0);\n"
        "    result[gid] = matrix[gid] - vector[gid];\n"
        "}\n";

// OpenCL kernel for Matrix-Vector Multiplication
const char *matrixVectorKernelSource =
        "__kernel void matrix_vector_multiply(__global const float *matrix, __global const float *vector, __global float *result, int rows, int cols) {\n"
        "    int gid = get_global_id(0);\n"
        "    float sum = 0.0f;\n"
        "    for (int j = 0; j < cols; ++j) {\n"
        "        sum += matrix[gid * cols + j] * vector[j];\n"
        "    }\n"
        "    result[gid] = sum;\n"
        "}\n";

// OpenCL kernel for Parallel 2-Norm Calculation
const char *parallelNormKernelSource =
        "__kernel void parallel_norm(__global const float *vector, __global float *result, int size) {\n"
        "    int gid = get_global_id(0);\n"
        "    result[gid] = vector[gid] * vector[gid];\n"
        "}\n";

// OpenCL kernel for Shifted Power Method
const char *shiftedPowerMethodKernelSource =
        "__kernel void shifted_power_method(__global const float *matrix, __global float *vector, int size, int iterations) {\n"
        "    for (int iter = 0; iter < iterations; ++iter) {\n"
        "        // Perform Shifted Power Method iteration\n"
        "        // Update 'vector' based on 'matrix'\n"
        "        float lambda = 0.0; // Choose the shift value here\n"
        "        float norm = 0.0;\n"
        "        for (int i = 0; i < size; ++i) {\n"
        "            float sum = 0.0;\n"
        "            for (int j = 0; j < size; ++j) {\n"
        "                sum += matrix[i * size + j] * vector[j];\n"
        "            }\n"
        "            vector[i] = sum - lambda * vector[i];\n"
        "            norm += vector[i] * vector[i];\n"
        "        }\n"
        "        norm = sqrt(norm);\n"
        "        for (int i = 0; i < size; ++i) {\n"
        "            vector[i] /= norm;\n"
        "        }\n"
        "    }\n"
        "}\n";

// OpenCL kernel for Power Method
const char *powerMethodKernelSource =
        "__kernel void power_method(__global const float *matrix, __global float *vector, int size, int iterations, __global float *lambda) {\n"
        "    for (int iter = 0; iter < iterations; ++iter) {\n"
        "        // Perform Power Method iteration\n"
        "        // Update 'vector' based on 'matrix'\n"
        "        float norm = 0.0;\n"
        "        for (int i = 0; i < size; ++i) {\n"
        "            float sum = 0.0;\n"
        "            for (int j = 0; j < size; ++j) {\n"
        "                sum += matrix[i * size + j] * vector[j];\n"
        "            }\n"
        "            vector[i] = sum;\n"
        "            norm += vector[i] * vector[i];\n"
        "        }\n"
        "        norm = sqrt(norm);\n"
        "        for (int i = 0; i < size; ++i) {\n"
        "            vector[i] /= norm;\n"
        "        }\n"
        "    }\n"
        "    *lambda = vector[0]; // Assign the first element of the 'vector' to 'lambda'\n"
        "}\n";

// OpenCL kernel for Matrix Subtraction
const char *matrixSubtractionKernelSource =
        "__kernel void matrix_subtraction(__global const float *matrixA, __global const float *matrixB, __global float *result, int rows, int cols) {\n"
        "    int gid = get_global_id(0);\n"
        "    for (int j = 0; j < cols; ++j) {\n"
        "        result[gid * cols + j] = matrixA[gid * cols + j] - matrixB[gid * cols + j];\n"
        "    }\n"
        "}\n";

void readMatrix(char *filename, float **matrix, int *rows, int *cols);

void writeVector(char *filename, float *vector, int size);

void checkError(cl_int status, const char *function);

void serialMatrixVectorMultiply(float *matrixA, float *vectorB, float *result, int rows, int cols);

double getCurrentTime();


int main() {
    // Read matrix A and vector B from file
    float *matrixA, *vectorB, *serialResult;
    int rows, cols;
    readMatrix("BigA.txt", &matrixA, &rows, &cols);


    vectorB = (float *) malloc(cols * sizeof(float));
    serialResult = (float *) malloc(rows * sizeof(float));
    for (int i = 0; i < cols; ++i) {
        vectorB[i] = 1.0;
    }
    cl_mem bufferResult;


    // Set up OpenCL
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);

    cl_uint maxComputeUnits;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, NULL);


    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, NULL);

    // Create OpenCL buffers for matrix A, vector B, and result vector
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, rows * cols * sizeof(float), NULL, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, cols * sizeof(float), NULL, NULL);
    bufferResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY, rows * sizeof(float), NULL, NULL);

    // Write data to OpenCL buffers
    clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, rows * cols * sizeof(float), matrixA, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, cols * sizeof(float), vectorB, 0, NULL, NULL);



    // Set work group size based on the number of compute units
    size_t localSize = maxComputeUnits > 1 ? maxComputeUnits : 1;

    // Set global size based on the number of compute units and local size
    size_t globalSize = localSize * (rows / localSize + (rows % localSize != 0 ? 1 : 0));
    // Serial Matrix-Vector Multiply
    double sequentialMatrixVectorMultiplyStartTime = getCurrentTime();
    serialMatrixVectorMultiply(matrixA, vectorB, serialResult, rows, cols);
    double sequentialMatrixVectorMultiplyEndTime = getCurrentTime();
    printf("Sequential Multiply Runtime: %.6f seconds\n",
           sequentialMatrixVectorMultiplyEndTime - sequentialMatrixVectorMultiplyStartTime);


    // Parallel Matrix-Vector Multiply
    double parallelMultiplyStartTime = getCurrentTime();
    matrixVectorMultiply(context, device, queue, bufferA, bufferB, bufferResult, rows, cols);
    double parallelMultiplyEndTime = getCurrentTime();
    printf("Parallel Multiply Runtime: %.6f seconds\n", parallelMultiplyEndTime - parallelMultiplyStartTime);

    // Read the result back to the host
    float *result = (float *) malloc(rows * sizeof(float));
    clEnqueueReadBuffer(queue, bufferResult, CL_TRUE, 0, rows * sizeof(float), result, 0, NULL, NULL);


    // Sequential Norm Calculation
    double sequentialNormStartTime = getCurrentTime();
    for (int i = 0; i < rows; ++i) {
        serialResult[i] = serialResult[i] * serialResult[i];
    }
    double sequentialNormEndTime = getCurrentTime();
    printf("Sequential Norm Runtime: %.6f seconds\n", sequentialNormEndTime - sequentialNormStartTime);


    // Parallel Norm Calculation
    double parallelNormStartTime = getCurrentTime();
    parallelNorm(context, device, queue, bufferResult, bufferResult, rows);
    double parallelNormEndTime = getCurrentTime();
    printf("Parallel Norm Runtime: %.6f seconds\n", parallelNormEndTime - parallelNormStartTime);


/*
    // Sequential Matrix Subtraction
    double sequentialSubtractionStartTime = getCurrentTime();
    for (int i = 0; i < rows * cols; ++i) {
        serialResult[i] = matrixA[i] - matrixB[i];
    }
    double sequentialSubtractionEndTime = getCurrentTime();
    printf("Sequential Subtraction Runtime: %.6f seconds\n", sequentialSubtractionEndTime - sequentialSubtractionStartTime);
*/
    // Parallel Matrix Subtraction
    double parallelSubtractionStartTime = getCurrentTime();
    matrixSubtraction(context, device, queue, bufferA, bufferA, bufferResult, rows, cols);
    double parallelSubtractionEndTime = getCurrentTime();
    printf("Parallel Subtraction Runtime: %.6f seconds\n", parallelSubtractionEndTime - parallelSubtractionStartTime);



    /*  // Calculate Speed-up and Efficiency
      double sequentialMatrixVectorMultiplyRuntime =
              sequentialMatrixVectorMultiplyEndTime - sequentialMatrixVectorMultiplyStartTime;
      double sequentialNormRuntime = sequentialNormEndTime - sequentialNormStartTime;
      double parallelMultiplyRuntime = parallelMultiplyEndTime - parallelMultiplyStartTime;
      double parallelNormRuntime = parallelNormEndTime - parallelNormStartTime;

      double parallelSubtractionRuntime = parallelSubtractionEndTime - parallelSubtractionStartTime;

      double speedUpMultiply = sequentialMatrixVectorMultiplyRuntime / parallelMultiplyRuntime;
      double speedUpNorm = sequentialNormRuntime / parallelNormRuntime;
      // double speedUpSubtraction = sequentialRuntime / parallelSubtractionRuntime;

      double efficiencyMultiply = speedUpMultiply / maxComputeUnits;
      double efficiencyNorm = speedUpNorm / maxComputeUnits;
      // double efficiencySubtraction = speedUpSubtraction / maxComputeUnits;

      printf("Speed-Up Multiply: %.6f\n", speedUpMultiply);
      printf("Efficiency Multiply: %.6f\n", efficiencyMultiply);

      printf("Speed-Up Norm: %.6f\n", speedUpNorm);
      printf("Efficiency Norm: %.6f\n", efficiencyNorm);

      //  printf("Speed-Up Subtraction: %.6f\n", speedUpSubtraction);
      //   printf("Efficiency Subtraction: %.6f\n", efficiencySubtraction);
  */



    // Read the result back to the host
    clEnqueueReadBuffer(queue, bufferResult, CL_TRUE, 0, rows * sizeof(float), result, 0, NULL, NULL);
// Power Method
    float lambda_opencl;
    double start_time_power_iteration = getCurrentTime();
    powerMethodOpenCL(context, device, queue, matrixA, vectorB, rows, 100, &lambda_opencl);
    double end_time_power_iteration = getCurrentTime();
    printf("\n");
    printf("Eigenvalue PowerMethod: %.6f\n", lambda_opencl);
    double parallel_power_iteration_run_time = ((double) (end_time_power_iteration - start_time_power_iteration));
    printf("Parallel Power Iteration Run Time: %.6f\n", parallel_power_iteration_run_time);


    // Sequential Power Iteration
    float lambda_sequential;
    double start_time_sequentialPowerIteration = getCurrentTime();
    sequentialPowerIteration(matrixA, vectorB, rows, 100, &lambda_sequential);
    double end_time_sequentialPowerIteration = getCurrentTime();
    double sequentialPowerIteration_run_time = ((double) (end_time_sequentialPowerIteration -
                                                          start_time_sequentialPowerIteration));
    printf("Sequential Power Iteration Run Time: %.6f\n", sequentialPowerIteration_run_time);

    // Shifted Power Method
    float lambda_shifted;
    double start_time_shiftedPowerMethod = getCurrentTime();
    shiftedPowerMethod(context, device, queue, matrixA, vectorB, bufferResult, rows, 100, &lambda_shifted);
    printf("Eigenvalue Shifted-Power Method: %.6f\n", lambda_shifted);
    double end_time_shiftedPowerMethod = getCurrentTime();
    double paralllel_shifted_power_method_run_time = ((double) (end_time_shiftedPowerMethod -
                                                                start_time_shiftedPowerMethod));
    printf("Parallel Shifted-Power Iteration Run Time: %.6f\n", paralllel_shifted_power_method_run_time);

    // Write the result vector to a file
    writeVector("eigenvector.txt", result, rows);

    // Calculate Speed-up and Efficiency
    printf("\nSpeed-up and Efficiency\n");
    printf("Speed-Up PowerIteration : %.6f\n", sequentialPowerIteration_run_time / parallel_power_iteration_run_time);
    printf("Efficiency PowerIteration: %.6f\n",
           (sequentialPowerIteration_run_time / parallel_power_iteration_run_time) / maxComputeUnits);


    // Clean up
    free(matrixA);
    free(vectorB);
    free(result);

    // Release OpenCL resources
    cleanupOpenCL(context, queue, bufferA, bufferB, bufferResult, NULL, NULL);

    return 0;
}

void serialMatrixVectorMultiply(float *matrixA, float *vectorB, float *result, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < cols; ++j) {
            result[i] += matrixA[i * cols + j] * vectorB[j];
        }
    }
}

double getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec * 1e-6;
}

void readMatrix(char *filename, float **matrix, int *rows, int *cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open matrix file.\n");
        exit(1);
    }

    fscanf(file, "%d %d", rows, cols);

    *matrix = (float *) malloc((*rows) * (*cols) * sizeof(float));
    if (*matrix == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            fscanf(file, "%f", &(*matrix)[i * (*cols) + j]);
        }
    }

    fclose(file);
}

void writeVector(char *filename, float *vector, int size) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Failed to open vector file.\n");
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        fprintf(file, "%f\n", vector[i]);
    }

    fclose(file);
}

void checkError(cl_int status, const char *function) {
    if (status != CL_SUCCESS) {
        fprintf(stderr, "Error during %s: %d\n", function, status);
        exit(1);
    }
}

void
matrixVectorMultiply(cl_context context, cl_device_id device, cl_command_queue queue, cl_mem bufferA, cl_mem bufferB,
                     cl_mem bufferResult, int rows, int cols) {
    // Create OpenCL program and kernel for matrix_vector_multiply
    cl_program program = clCreateProgramWithSource(context, 1, &matrixVectorKernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "matrix_vector_multiply", NULL);

    // Set OpenCL kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferResult);
    clSetKernelArg(kernel, 3, sizeof(int), &rows);
    clSetKernelArg(kernel, 4, sizeof(int), &cols);

    // Determine the number of compute units
    cl_uint maxComputeUnits;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, NULL);

    // Set work group size based on the number of compute units
    size_t localSize = maxComputeUnits > 1 ? maxComputeUnits : 1;

    // Set global size based on the number of compute units and local size
    size_t globalSize = localSize * (rows / localSize + (rows % localSize != 0 ? 1 : 0));

    // Execute OpenCL kernel
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    clFinish(queue);
    //  printf("maxComputeUnitsmultpply: %d",maxComputeUnits);
    // Read the result back to the host
    float *result = (float *) malloc(rows * sizeof(float));
    clEnqueueReadBuffer(queue, bufferResult, CL_TRUE, 0, rows * sizeof(float), result, 0, NULL, NULL);

    // Clean up OpenCL resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}


void
parallelNorm(cl_context context, cl_device_id device, cl_command_queue queue, cl_mem bufferVector, cl_mem bufferResult,
             int size) {
    // Create OpenCL program and kernel for parallel_norm
    cl_program program = clCreateProgramWithSource(context, 1, &parallelNormKernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "parallel_norm", NULL);

    // Set OpenCL kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferVector);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferResult);
    clSetKernelArg(kernel, 2, sizeof(int), &size);

    // Determine the number of compute units
    cl_uint maxComputeUnits;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, NULL);

    // Set work group size based on the number of compute units
    size_t localSize = maxComputeUnits > 1 ? maxComputeUnits : 1;

    // Set global size based on the number of compute units and local size
    size_t globalSize = localSize * (size / localSize + (size % localSize != 0 ? 1 : 0));

    // Execute OpenCL kernel
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    clFinish(queue);
//    printf("maxComputeUnitsparallelNorm: %d",maxComputeUnits);
    // Clean up OpenCL resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

void shiftedPowerMethod(cl_context context, cl_device_id device, cl_command_queue queue, float *matrixA, float *vectorB,
                        cl_mem bufferVector, size_t size, int iterations, float *lambda) {
    // Create OpenCL program and kernel for shifted_power_method
    cl_program program = clCreateProgramWithSource(context, 1, &shiftedPowerMethodKernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "shifted_power_method", NULL);

    // Create OpenCL buffers for matrix and vector
    cl_mem bufferMatrix = clCreateBuffer(context, CL_MEM_READ_ONLY, size * size * sizeof(float), NULL, NULL);

    // Write data to OpenCL buffers
    clEnqueueWriteBuffer(queue, bufferMatrix, CL_TRUE, 0, size * size * sizeof(float), matrixA, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufferVector, CL_TRUE, 0, size * sizeof(float), vectorB, 0, NULL, NULL);

    // Set OpenCL kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferMatrix);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferVector);
    clSetKernelArg(kernel, 2, sizeof(int), &size);
    clSetKernelArg(kernel, 3, sizeof(int), &iterations);

    // Execute OpenCL kernel in a loop for the specified number of iterations
    for (int iter = 0; iter < iterations; ++iter) {
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &size, NULL, 0, NULL, NULL);
        clFinish(queue);
    }

    // Read the result back to the host
    clEnqueueReadBuffer(queue, bufferVector, CL_TRUE, 0, size * sizeof(float), vectorB, 0, NULL, NULL);

    float sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += vectorB[i] * vectorB[i];
    }

    *lambda = sum;

    // Clean up OpenCL resources
    clReleaseMemObject(bufferMatrix);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

void matrixSubtraction(cl_context context, cl_device_id device, cl_command_queue queue, cl_mem bufferMatrixA,
                       cl_mem bufferMatrixB, cl_mem bufferResult, int rows, int cols) {
    // Create OpenCL program and kernel for matrix_subtraction
    cl_program program = clCreateProgramWithSource(context, 1, &matrixSubtractionKernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "matrix_subtraction", NULL);

    // Set OpenCL kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferMatrixA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferMatrixB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferResult);
    clSetKernelArg(kernel, 3, sizeof(int), &rows);
    clSetKernelArg(kernel, 4, sizeof(int), &cols);

    // Determine the number of compute units
    cl_uint maxComputeUnits;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, NULL);

    // Set work group size based on the number of compute units
    size_t localSize = maxComputeUnits > 1 ? maxComputeUnits : 1;

    // Set global size based on the number of compute units and local size
    size_t globalSize = localSize * (rows / localSize + (rows % localSize != 0 ? 1 : 0));

    // Execute OpenCL kernel
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    clFinish(queue);

    // Read the result back to the host
    clEnqueueReadBuffer(queue, bufferResult, CL_TRUE, 0, rows * cols * sizeof(float), bufferResult, 0, NULL, NULL);

    // Clean up OpenCL resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

// Function for Power Method using OpenCL
void powerMethodOpenCL(cl_context context, cl_device_id device, cl_command_queue queue, float *matrixA, float *vectorB,
                       size_t size, int iterations, float *lambda) {
    // Create OpenCL program and kernel for power_method
    cl_program program = clCreateProgramWithSource(context, 1, &powerMethodKernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "power_method", NULL);

    // Create OpenCL buffers for matrix and vector
    cl_mem bufferMatrix = clCreateBuffer(context, CL_MEM_READ_ONLY, size * size * sizeof(float), NULL, NULL);
    cl_mem bufferVector = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(float), NULL, NULL);

    // Write data to OpenCL buffers
    clEnqueueWriteBuffer(queue, bufferMatrix, CL_TRUE, 0, size * size * sizeof(float), matrixA, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufferVector, CL_TRUE, 0, size * sizeof(float), vectorB, 0, NULL, NULL);

    // Set OpenCL kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferMatrix);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferVector);
    clSetKernelArg(kernel, 2, sizeof(int), &size);
    clSetKernelArg(kernel, 3, sizeof(int), &iterations);

    // Execute OpenCL kernel in a loop for the specified number of iterations
    for (int iter = 0; iter < iterations; ++iter) {
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &size, NULL, 0, NULL, NULL);
        clFinish(queue);
    }

    // Read the result back to the host
    clEnqueueReadBuffer(queue, bufferVector, CL_TRUE, 0, size * sizeof(float), vectorB, 0, NULL, NULL);

    float sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += vectorB[i] * vectorB[i];
    }

    *lambda = sum;

    // Clean up OpenCL resources
    clReleaseMemObject(bufferMatrix);
    clReleaseMemObject(bufferVector);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

void matrixVectorSubtraction(cl_context context, cl_device_id device, cl_command_queue queue, cl_mem bufferMatrix,
                             cl_mem bufferVector, cl_mem bufferResult, int size) {
    // Create OpenCL program and kernel for matrix_vector_subtraction
    cl_program program = clCreateProgramWithSource(context, 1, &matrixVectorSubtractionKernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "matrix_vector_subtraction", NULL);

    // Set OpenCL kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferMatrix);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferVector);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferResult);
    clSetKernelArg(kernel, 3, sizeof(int), &size);

    // Determine the number of compute units
    cl_uint maxComputeUnits;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, NULL);

    // Set work group size based on the number of compute units
    size_t localSize = maxComputeUnits > 1 ? maxComputeUnits : 1;

    // Set global size based on the number of compute units and local size
    size_t globalSize = localSize * (size / localSize + (size % localSize != 0 ? 1 : 0));

    // Execute OpenCL kernel
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    clFinish(queue);

    // Clean up OpenCL resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

void cleanupOpenCL(cl_context context, cl_command_queue queue, cl_mem bufferA, cl_mem bufferB, cl_mem bufferResult,
                   cl_kernel kernel, cl_program program) {
    // Release OpenCL resources
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferResult);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void sequentialPowerIteration(float *matrixA, float *vectorB, int size, int iterations, float *lambda) {
    // Allocate memory for intermediate vectors
    float *tempVector = (float *) malloc(size * sizeof(float));

    for (int iter = 0; iter < iterations; ++iter) {
        // Matrix-vector multiplication: tempVector = matrixA * vectorB
        for (int i = 0; i < size; ++i) {
            tempVector[i] = 0.0;
            for (int j = 0; j < size; ++j) {
                tempVector[i] += matrixA[i * size + j] * vectorB[j];
            }
        }

        // Find the 2-norm of the resulting vector
        float norm = 0.0;
        for (int i = 0; i < size; ++i) {
            norm += tempVector[i] * tempVector[i];
        }
        norm = sqrt(norm);

        // Normalize the vector
        for (int i = 0; i < size; ++i) {
            vectorB[i] = tempVector[i] / norm;
        }
    }

    // Calculate the eigenvalue
    *lambda = vectorB[0];

    // Free allocated memory
    free(tempVector);
}

void sequentialShiftedPowerIteration(float *matrixA, float *vectorB, int size, int iterations, float *eigenvalue) {
    // Allocate memory for temporary vectors
    float *tempVector = (float *) malloc(size * sizeof(float));
    float *tempResult = (float *) malloc(size * sizeof(float));

    // Initialize the initial vector for power iteration
    for (int i = 0; i < size; ++i) {
        vectorB[i] = 1.0;
    }

    // Perform shifted power iteration
    for (int iter = 0; iter < iterations; ++iter) {

        for (int i = 0; i < size; ++i) {
            tempVector[i] = 0.0;
            for (int j = 0; j < size; ++j) {
                tempVector[i] += matrixA[i * size + j] * vectorB[j];
            }
        }


        for (int i = 0; i < size; ++i) {
            matrixA[i * size + i] -= *eigenvalue;
        }


        for (int i = 0; i < size; ++i) {
            tempResult[i] = 0.0;
            for (int j = 0; j < size; ++j) {
                tempResult[i] += matrixA[i * size + j] * tempVector[j];
            }
        }

        // Normalize the result
        float norm = 0.0;
        for (int i = 0; i < size; ++i) {
            norm += tempResult[i] * tempResult[i];
        }
        norm = sqrt(norm);

        // Update the eigenvalue
        *eigenvalue = tempResult[0] / vectorB[0];

        // Normalize the vector for the next iteration
        for (int i = 0; i < size; ++i) {
            vectorB[i] = tempResult[i] / norm;
        }

        // Shift the matrix back to the original form
        for (int i = 0; i < size; ++i) {
            matrixA[i * size + i] += *eigenvalue;
        }
    }

    // Clean up
    free(tempVector);
    free(tempResult);
}