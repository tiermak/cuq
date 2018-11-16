# cuq
CUDA multi-GPU concurrent tasks queue

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
