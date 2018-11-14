#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <cuq.h>

#define THREADS_PER_BLOCK 64

using namespace std;

__global__
void vectorAdd(float * a, float * b, float * c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  c[i] = a[i] + b[i];
}

int done[8];

//Define GPU task by inheriting from GPUTask
//In fact it should hold data for calculations and code for calculations defined in doWork() method
class VectorAddTask: public GPUTask {
  public:
    //constructor can be arbitrary
    VectorAddTask(float * _h_a, float * _h_b, float * _h_c, int _size, int _id) {
      id = _id;
      size = _size;
      h_a = _h_a;
      h_b = _h_b;
      h_c = _h_c;
    }

    //All GPU calculations should be done in this method
    void doWork() {
      int device;
      cudaGetDevice(&device);
      cout << "Device: " << device << ", running task: " << id << ", size: " << size << endl;

      float * d_a;
      cudaMalloc(&d_a, size * size * sizeof(float));
      cudaMemcpy(d_a, h_a, size * size * sizeof(float), cudaMemcpyHostToDevice);

      float * d_b;
      cudaMalloc(&d_b, size * size * sizeof(float));
      cudaMemcpy(d_b, h_b, size * size * sizeof(float), cudaMemcpyHostToDevice);
      
      float * d_c;
      cudaMalloc(&d_c, size * size * sizeof(float));

      int blocksCount = (int)ceil((float)size * size / THREADS_PER_BLOCK);
      for (int i = 0; i < 1024; i++) {
        vectorAdd<<<blocksCount,THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
      }

      cudaMemcpy(h_c, d_c, size * size * sizeof(float), cudaMemcpyDeviceToHost);
      
      cudaFree(d_a);
      cudaFree(d_b);
      cudaFree(d_c);

      cout << "Device: " << device << ", task: " << id << " finished" << ", size: " << size << endl;
      //increase number of finished tasks
      done[device] += 1;
    }

    //Destructor is empty in this case
    ~VectorAddTask() {
    }
  
  private:
    int size;
    int id;
    float * h_a;
    float * h_b;
    float * h_c;
};

int pow(int a, int b) {
  int res = 1;
  for (int i = 0; i < b; i++)
    res *= a;

  return res;
}

int main() {
  int devices = 8;
  int tasksCount = 128 * 128;
  int size = 4096;

  float * h_a = new float[size * size];
  float * h_b = new float[size * size];
  float * h_c = new float[size * size];

  for (int i = 0; i < size * size; i++) {
    h_a[i] = i;
    h_b[i] = i + 100500;
  }

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int> dist (0,12);

  GPUTask ** tasks = new GPUTask *[tasksCount];
  for (int i = 0; i < tasksCount; i++) {
    //randomize size of task
    int randSize = size / pow(2, dist(mt));
    tasks[i] = new VectorAddTask(h_a, h_b, h_c, randSize, i);
  }

  for (int i = 0; i < 8; i++) {
    done[i] = 0;
  }

  processTasks(tasks, tasksCount, devices);

  //number of finished tasks per device should be more or less equal
  for (int i = 0; i < 8; i++) {
    cout << "Device: " << i << ", done: " << done[i] << endl;
  }

  for (int i = 0; i < tasksCount; i++) {
    delete tasks[i];
  }
  delete[] tasks;
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;

  return 0;
}
