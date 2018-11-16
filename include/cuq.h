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
    bool processNext();
    GPUTasksQueue(GPUTask** tasks, int tasksCount);
  private:
    std::mutex queueMutex;
    std::queue<GPUTask*> tasksQueue;
};

extern "C"
void processTasks(GPUTask** tasks, int taskCount, int devicesCount);
