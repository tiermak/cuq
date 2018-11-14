#include <mutex>
#include <queue>

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

void processTasks(GPUTask** tasks, int taskCount, int devicesCount);
