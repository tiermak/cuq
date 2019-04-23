#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unistd.h>

#include <nvml.h>
#include <cuda_runtime.h>

#define NVML(x)  { nvmlReturn_t ret = x; if (ret != NVML_SUCCESS) { printf("NVML error in line %i: %s\n", __LINE__, nvmlErrorString(ret)); \
exit(8); } }

#define gpuErrchk(code, device, errorMsg) { \
  if (code != cudaSuccess) { \
    string msg = "Error while trying to occupy device " + to_string(device) + ". " + string(cudaGetErrorString(code)); \
    memcpy(errorMsg, msg.c_str(), msg.length()); \
    return -1; \
  } \
}

inline void gpuAssert(cudaError_t code, const char *file, int line) {

}

using namespace std;

vector<int> getAllPhysicallyAvailableDevices() {

  unsigned int i, n;
  NVML( nvmlInit() );
  NVML( nvmlDeviceGetCount(&n) );
  if (n > 128) {
    string msg = "nvmlDeviceGetCount returned " +  to_string(n) + ". This does not make sense.";
    throw std::runtime_error(msg);
  }

  vector<int> res;

  int currentPid = getpid();

  for (i = 0; i < n; i++) {
    nvmlDevice_t dev;
    unsigned int num_procs = 128;
    nvmlProcessInfo_t procs[128];
    NVML( nvmlDeviceGetHandleByIndex(i, &dev) );
    NVML( nvmlDeviceGetComputeRunningProcesses(dev, &num_procs, procs) );
    if (num_procs < 1) {
      res.push_back(i);
      // cout << "getAllPhysicallyAvailableDevices:, i: " << i << endl;
    } else {
      //also add deviceIdx if current process occupied this device already
      for (int p = 0; p < num_procs; p++){
        if (procs[p].pid == currentPid)
          res.push_back(i);
      }
    }
  }

  return res;
}

vector<int> readCudaVisibleDevices() {
  vector<int> res;

  auto rawValue = std::getenv("CUDA_VISIBLE_DEVICES");

  if (!rawValue)
    return res;

  string visibleDevices = string(rawValue);
  stringstream ss(visibleDevices);

  if (ss.peek() == '"' || ss.peek() == '\'')
    ss.ignore();

  int i;
  while (ss >> i){
    res.push_back(i);

    // cout << "readCudaVisibleDevices:, i: " << i << endl;

    if (ss.peek() == ',' || ss.peek() == '"' || ss.peek() == '\'')
      ss.ignore();
  }

  return res;
}

vector<int> getAvailableDevices() {
  vector<int> visibleDevices = readCudaVisibleDevices();

  if (visibleDevices.empty())
    return getAllPhysicallyAvailableDevices();

  return visibleDevices;
}

extern "C"
int occupyDevices(int requestedDevicesCount, int * occupiedDevicesIdxs, char * errorMsg) {
  try {
    vector<int> availableDevcices = getAvailableDevices();

    if ((int)availableDevcices.size() < requestedDevicesCount) {
      string msg = "There are not as many free devices as requested. Requested devices count: " 
        + to_string(requestedDevicesCount) + ". Available devices count: " + to_string(availableDevcices.size()) + ".";
      
      memcpy(errorMsg, msg.c_str(), msg.length());

      return -1;
    }

    int nextDeviceIdx = 0;
    for (int i = 0; i < requestedDevicesCount; i++) {
      int deviceIdx = availableDevcices[i];
      gpuErrchk( cudaSetDevice(i), deviceIdx, errorMsg );

      //call some API functions to really occupy device (I'm lazy to look for more elegant way to do it)
      char * ddata;
      gpuErrchk( cudaMalloc(&ddata, 1), deviceIdx, errorMsg );
      gpuErrchk( cudaFree(ddata), deviceIdx, errorMsg );
      
      occupiedDevicesIdxs[nextDeviceIdx++] = deviceIdx;
    }
  } catch (const std::exception& e) {
    auto msg = string(e.what());
    memcpy(errorMsg, msg.c_str(), msg.length());
    return -1;
  }

  return 0;
}
