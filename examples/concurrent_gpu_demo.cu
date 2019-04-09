#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <cuq.h>

#define THREADS_PER_BLOCK 64

using namespace std;

__global__
void vectorAdd(float * a, float * b, float * c, int iterations) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  for (int j = 0; j < iterations; j++)
    c[i] = a[i] + b[i];
}

int done[8];
int SIZE = 16384;

//Define GPU task by inheriting from GPUTask
//In fact it should hold data for calculations and code for calculations defined in doWork() method
class VectorAddTask: public GPUTask {
  public:
    //constructor can be arbitrary
    VectorAddTask(float * _h_a, float * _h_b, float * _h_c, int _iterations, int _id) {
      id = _id;
      iterations = _iterations;
      h_a = _h_a;
      h_b = _h_b;
      h_c = _h_c;
    }

    //All GPU calculations should be done in this method
    void doWork() {
      int device;
      cudaGetDevice(&device);
      cout << "Device: " << device << ", running task: " << id << ", iterations: " << iterations << endl;

      cudaMalloc(&d_a, SIZE * sizeof(float));
      cudaMemcpy(d_a, h_a, SIZE * sizeof(float), cudaMemcpyHostToDevice);

      cudaMalloc(&d_b, SIZE * sizeof(float));
      cudaMemcpy(d_b, h_b, SIZE * sizeof(float), cudaMemcpyHostToDevice);
      
      cudaMalloc(&d_c, SIZE * sizeof(float));

      int blocksCount = (int)ceil((float)SIZE / THREADS_PER_BLOCK);
      for (int i = 0; i < 1024; i++) {
        vectorAdd<<<blocksCount,THREADS_PER_BLOCK>>>(d_a, d_b, d_c, iterations);
      }

      cudaMemcpy(h_c, d_c, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

      cout << "Device: " << device << ", task: " << id << " finished" << ", iterations: " << iterations << endl;
      //increase number of finished tasks
      done[device] += 1;

      cudaFree(d_a);
      cudaFree(d_b);
      cudaFree(d_c);
    }

    //Destructor is empty in this case
    ~VectorAddTask() {
    }
  
  private:
    int iterations;
    int id;
    float * h_a;
    float * h_b;
    float * h_c;
    
    float * d_a;
    float * d_b;
    float * d_c;
};

int pow(int a, int b) {
  int res = 1;
  for (int i = 0; i < b; i++)
    res *= a;

  return res;
}

int main(int argc, char *argv[]) {
  int devicesCount;
  if (argc <= 1)
    devicesCount = 1;
  else 
    devicesCount = std::stoi(argv[1]);
  
  int * devices = new int[devicesCount];
  for (int d = 0; d < devicesCount; d++)
    devices[d] = d;

  cout << "cuq demo on " << devicesCount << " devices..." << endl;

  int tasksCount = 4096;

  float * h_a = new float[SIZE];
  float * h_b = new float[SIZE];
  float * h_c = new float[SIZE];

  for (int i = 0; i < SIZE; i++) {
    h_a[i] = i;
    h_b[i] = i + 100500;
  }

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int> dist (0,12);

  GPUTask ** tasks = new GPUTask *[tasksCount];
  for (int i = 0; i < tasksCount; i++) {
    //randomize interations number of task
    int randSize = SIZE / pow(2, dist(mt));
    tasks[i] = new VectorAddTask(h_a, h_b, h_c, randSize, i);
  }

  for (int i = 0; i < 8; i++) {
    done[i] = 0;
  }

  processTasksOnDevices(tasks, tasksCount, devices, devicesCount, /*resetDeviceAfterFinish =*/ true, /*deleteTasksAutomatically =*/ true);

  //number of finished tasks per device should be more or less equal
  for (int i = 0; i < devicesCount; i++) {
    cout << "Device: " << i << ", done: " << done[i] << endl;
  }

  delete[] devices;
  delete[] tasks;
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;

  return 0;
}
