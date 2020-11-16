#pragma once

#include <mutex>
#include <queue>
#include <functional>


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
 
  int calculationsCount(int localI, int groupSize, int wholeSize) {
    int startingIndex = localI * groupSize;

    if (startingIndex < wholeSize - groupSize) //we are not close to an end of our data
      return groupSize;

    int rest = wholeSize - startingIndex;

    if (rest >= 0) //the last chunk of the data
      return rest;

    return 0; //the data ended, nothing more to process
  }

  private:
    T params;
    std::function<void(T)> gpuCalculations;
};

int calculationsCount(int localI, int groupSize, int wholeSize);

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

