#include <mutex>
#include <queue>

#pragma once

class GPUTask {
  public:
    virtual void doWork() = 0;
    virtual ~GPUTask();
};

class GPUTasksQueue {
  public:
    bool resetDeviceAfterFinish;
    bool deleteTasksAutomatically;
    bool processNext();
    GPUTasksQueue(GPUTask** tasks, int tasksCount, bool _resetDeviceAfterFinish = true, bool _deleteTasksAutomatically = false);
  private:
    std::mutex queueMutex;
    std::queue<GPUTask*> tasksQueue;
};

extern "C"
void processTasksOnDevices(
  GPUTask ** tasks, int taskCount, 
  int * devices, int devicesCount, 
  bool resetDeviceAfterFinish = true, bool deleteTasksAutomatically = false);

extern "C"
void deleteTasks(GPUTask** tasks, int taskCount);

extern "C"
void processTasks(
  GPUTask ** tasks, int taskCount,
  int requestedDevicesCount, 
  bool deleteTasksAutomatically = false);

extern "C"
int occupyDevices(int requestedDevicesCount, int * occupiedDevicesIdxs, char * errorMsg);

