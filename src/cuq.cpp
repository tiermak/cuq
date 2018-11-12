#include <thread>
#include <cuda_runtime.h>
#include <cuq.h>

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

  return true;
}

GPUTasksQueue::GPUTasksQueue(GPUTask ** tasks, int tasksCount) {
  for (int i = 0; i < tasksCount; i++)
    tasksQueue.push(tasks[i]);
}

void threadStart(GPUTasksQueue *queue, int device) {
  //assign current thread to device
  cudaSetDevice(device);
  
  //pull tasks until all of them are finished
  while(queue->processNext()) {};

  //release device
  cudaDeviceReset();
}

void processTasks(GPUTask ** tasks, int taskCount, int devicesCount) {
  //create a queue of GPU tasks (which is thread safe internally)
  GPUTasksQueue *queue = new GPUTasksQueue(tasks, taskCount);

  std::thread * threads = new std::thread[devicesCount];

  //start one thread per device
  for (int d = 0; d < devicesCount; d++)
    threads[d] = std::thread(threadStart, queue, d);

  //wait all threads to finish
  for (int d = 0; d < devicesCount; d++)
    threads[d].join();

  delete[] threads;
  delete queue;
}
