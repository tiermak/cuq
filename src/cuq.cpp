#include <thread>
#include <cuda_runtime.h>
#include <cuq.h>
#include <iostream>
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>

using namespace std;

GPUTask::~GPUTask() {
}

bool GPUTasksQueue::processNext() {
  queueMutex.lock();
  
  if (tasksQueue.empty()) {
    queueMutex.unlock();
    return false;
  }

  GPUTask * nextTask = tasksQueue.front();
  tasksQueue.pop();
  
  queueMutex.unlock();
  
  nextTask->doWork();

  if (deleteTasksAutomatically)
    delete nextTask;

  return true;
}

GPUTasksQueue::GPUTasksQueue(GPUTask ** tasks, int tasksCount, bool _resetDeviceAfterFinish, bool _deleteTasksAutomatically) {
  for (int i = 0; i < tasksCount; i++)
    tasksQueue.push(tasks[i]);
  
  resetDeviceAfterFinish = _resetDeviceAfterFinish;
  deleteTasksAutomatically = _deleteTasksAutomatically;
}

void threadStart(GPUTasksQueue *queue, int device) {
  //assign current thread to device
  cudaSetDevice(device);

  //pull tasks until all of them are finished
  while(queue->processNext()) {};

  //release device
  if (queue->resetDeviceAfterFinish)
    cudaDeviceReset();
}

inline void printStackTrace(string msg) {
  void *array[20];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 20);

  // print out all the frames to stderr
  cerr << msg << endl;
  backtrace_symbols_fd(&array[1], size - 1, STDERR_FILENO);
}

void signalHandler(int sig) {
  auto msg = "Error: signal " + to_string(sig);
  printStackTrace(msg);
  exit(EXIT_FAILURE);
}

extern "C"
void processTasksOnDevices(
  GPUTask ** tasks, int taskCount, 
  int * devices, int devicesCount, 
  bool resetDeviceAfterFinish, bool deleteTasksAutomatically) {
    
  signal(SIGSEGV, signalHandler);

  //create a queue of GPU tasks (which is thread safe internally)
  GPUTasksQueue *queue = new GPUTasksQueue(tasks, taskCount, resetDeviceAfterFinish, deleteTasksAutomatically);

  std::thread * threads = new std::thread[devicesCount];

  //start one thread per device
  for (int d = 0; d < devicesCount; d++) {
    int device = devices[d];
    threads[d] = std::thread(threadStart, queue, device);
  }

  //wait all threads to finish
  for (int d = 0; d < devicesCount; d++)
    threads[d].join();

  delete[] threads;
  delete queue;
}

extern "C"
void deleteTasks(GPUTask** tasks, int taskCount) {
  for (int i = 0; i < taskCount; i++)
    delete tasks[i];
}

extern "C"
void processTasks(
  GPUTask ** tasks, int taskCount,
  int requestedDevicesCount, 
  bool resetDeviceAfterFinish, bool deleteTasksAutomatically) {
  
  char errorMsg[1000];
  int devices[128];

  int res = occupyDevices(requestedDevicesCount, devices, errorMsg);

  if (res == 0) {
    processTasksOnDevices(tasks, taskCount, devices, requestedDevicesCount, resetDeviceAfterFinish, deleteTasksAutomatically);
  } else {
    std::cerr << errorMsg;
  }
}
