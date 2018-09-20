//
//  изображение размером 6498 x 3890
// 734.038 мсек — Скорость на видеокарте
// 547.168 мсек скорость на процессоре параллельно
// 64.3 мсек — скорость на процессоре
// размер изображения 670 x 465
// 0.669 мсек - скорость на процессоре
// 10.245 мсек - скорость на видеокарте
// 6.8 мсек - параллельно на процессоре
//

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

void inCP(cv::Mat image) {
    std::chrono::high_resolution_clock::time_point tStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols/2; j++) {
            auto buf = image.at<cv::Vec3b>(i, j);
            image.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(i, image.cols-j);
                                                            image.at<cv::Vec3b>(i, image.cols-j) = buf;
        }
    }
    std::chrono::high_resolution_clock::time_point tEnd = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();
    std::cout << "Время на ЦП: " << (double)duration / 1000 << std::endl;
}

int main(int argc, char** argv)
{
    cv::Mat img = cv::imread(argv[3]);
    inCP(img);
    int err;
    char *KernelSource = (char*) malloc(1000000); // указатель на буфер со строкой - кодом kernel-функции

    size_t global;                      // размер грида ()в потоках)
    size_t local;                       // размер блока потоков

    cl_device_id device_id;             // id вычислителя
    cl_context context;                 // контекст
    cl_command_queue commands;          // очередь заданий для выполнения на видеокарте
    cl_program program;                 // наша программа для видеокарты (она сейчас состоит всего из одной функции)
    cl_kernel kernel;                   // объект, соответствующий нашей kernel-функции

    cl_mem input;                       // буфер для входных данных на видеокарте
    
    // ищем вычислительное устройство нужного типа
    int gpu = 0;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }

    // создаем контекст
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // создаем очередь задач
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }
    
    // читаем код kernel-функции из файла
    FILE *fp = fopen(argv[1], "r");
    if (fp == NULL) {
        printf("Error: Не могу прочесть файл с кодом kernel-функции!\n");
        return EXIT_FAILURE;
    }
    char symbol;
    int i=0;
    while((symbol = getc(fp)) != EOF) {
        KernelSource[i++] = symbol;
    }
    fclose(fp);
    KernelSource[i] = 0;
//    printf("%s", KernelSource);
    
    // создаем объект-программу для видеокарты
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // компилируем программу для видеокарты
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // находим в программе для видеокарты точку входа - kernel функию с нужным именем
    kernel = clCreateKernel(program, "square", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // выделяем память на видеокарте для входных данных и результата
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(uchar) * img.rows * img.cols * 3, NULL, NULL);
    if (!input)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }

    // копируем входные данные на видеокарту
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(uchar) *img.rows * img.cols * 3, img.data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    // задаем параметры вызова kernel-функции
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(unsigned int), &img.rows);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &img.cols);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // определяем максимальные размер блока потоков
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // запускаем нашу kernel-функцию на гриде из count потоков с найденным максимальным размером блока
    global = ((img.rows * img.cols * 3) + (local - 1)) / local * local;
    cl_event event;
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, &event);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }
    
    // ждем завершения выполнения задачи
    err = clWaitForEvents(1, &event);
//    clFinish(commands);
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    if (err)
    {
      printf("Error: Failed to execute kernel!\n");
      return EXIT_FAILURE;
    }
//    cl_ulong time_start, time_end;
//    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
//    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
//    if (gpu == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        std::cout << "Время на ЦП параллельно: " << (double)duration / 1000 << std::endl;
//    }
//    else std::cout << "время на видеокарте: " << (double)(time_end - time_start)/1e9 << std::endl;
    clReleaseEvent(event);
    
    // копируем результаты с видеокарты
    err = clEnqueueReadBuffer( commands, input, CL_TRUE, 0, sizeof(uchar) * img.rows * img.cols * 3, img.data, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    cv::imwrite("./result.jpg", img);
    printf("я всё.\n");
    
    // освобождаем память
    clReleaseMemObject(input);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}

