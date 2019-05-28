#include <mutex>
#include <queue>
#include <functional>

#pragma once

class GPUTask {
  public:
    virtual void doWork() = 0;
    virtual ~GPUTask();
};

template <class T>
class LambdaGPUTask : public GPUTask {
  public:
    LambdaGPUTask(T _params, std::function<void(T)> _gpuCalculations) {
      params = _params;
      gpuCalculations = _gpuCalculations;
    }

    void doWork() {
      gpuCalculations(params);
    }

  private:
    T params;
    std::function<void(T)> gpuCalculations;
};

class GPUTasksQueue {
  public:
    bool resetDeviceAfterFinish;
    bool deleteTasksAutomatically;
    bool processNext();
    GPUTasksQueue(GPUTask** tasks, int tasksCount, bool _resetDeviceAfterFinish = false, bool _deleteTasksAutomatically = true);
  private:
    std::mutex queueMutex;
    std::queue<GPUTask*> tasksQueue;
};

extern "C"
void deleteTasks(GPUTask** tasks, int taskCount);

extern "C"
void processTasks(
  GPUTask ** tasks, int taskCount,
  int requestedDevicesCount, 
  bool resetDeviceAfterFinish = false, bool deleteTasksAutomatically = true, bool handleSignals = true);

extern "C"
int occupyDevices(int requestedDevicesCount, int * occupiedDevicesIdxs, char * errorMsg);

