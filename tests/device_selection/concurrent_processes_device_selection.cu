#include <iostream>
#include <unistd.h>
#include <sys/wait.h>
#include <cuq.h>

using namespace std;

class IdleTask: public GPUTask {
  public: 
    unsigned int idleMicros;

    IdleTask(unsigned int _idleMicros) {
      idleMicros = _idleMicros;
    }

    void doWork() {
      usleep(idleMicros);
    }
};

int main(int argc, char * argv[]) {
  int processesCount;
  if (argc <= 1)
    processesCount = 1;
  else 
    processesCount = std::stoi(argv[1]);

  if (processesCount > 8) {
    cout << "There is no actual need to run more than 8 concurrent processes for this test..." << endl;
    return 0;
  }

  int parentPid = getpid();

  cout << "Parent PID: " << parentPid << endl;

  int TASKS_COUNT = 10;

  for (int p = 0; p < processesCount; p++){
    if (getpid() == parentPid) {
      if (!fork()) {

        auto tasks = new GPUTask*[TASKS_COUNT];
        for (int i = 0; i < TASKS_COUNT; i++) {
          tasks[i] = new IdleTask(i * 1000 * 1000);
        }

        cout << "PID: " << getpid() << ", run tasks..." << endl;

        processTasks(tasks, TASKS_COUNT, 1, true, true);

        delete tasks;

        cout << "PID: " << getpid() << ", tasks finished..." << endl;
      }
    }
  }

  if (getpid() == parentPid)
    wait(NULL);
  
  return 0;
}