# cuq
CUDA multi-GPU concurrent tasks queue

## Build
Requrements:
* CMake of version 3.10 or higher

```bash
mkdir build && cd build
cmake ..
make -j 8
[sudo] make install # to install library into lib directory and header into include (may require sudo)
```

## Example
```c++
int N = 42;

//create some tasks
GPUTask * tasks[N];
for (int i = 0; i < N; i++) 
  tasks[i] = createTask(...); //create task somehow

int devicesCount = 8;

//run all tasks on 8 devices
processTasks(tasks, N, devicesCount);

//then get results back from tasks
...

//delete tasks after all calculations are done
deleteTasks(tasks, N);
```

## Working demo
[examples/concurrent_gpu_demo.cu](examples/concurrent_gpu_demo.cu)
