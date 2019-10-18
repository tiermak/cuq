#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <utility>
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

pair<vector<int>, bool> getAvailableDevices() {
  vector<int> visibleDevices = readCudaVisibleDevices();

  // cout << ">>>>> visibleDevices.empty()" << visibleDevices.empty() << endl; 

  if (visibleDevices.empty())
    return make_pair(getAllPhysicallyAvailableDevices(), false);

  return make_pair(visibleDevices, true);
}

extern "C"
int occupyDevices(int requestedDevicesCount, int * occupiedDevicesIdxs, char * errorMsgOut) {
  try {
    auto availableDevcicesPair = getAvailableDevices();
    auto availableDevices = availableDevcicesPair.first;
    bool cudaVisibleDevciesSetProperly = availableDevcicesPair.second;

    if ((int)availableDevices.size() < requestedDevicesCount) {
      string msg = "There are not as many free devices as requested. Requested devices count: " 
        + to_string(requestedDevicesCount) + ". Available devices count: " + to_string(availableDevices.size()) + ".";
      
      memcpy(errorMsgOut, msg.c_str(), msg.length());

      return -1;
    }

    int nextDeviceIdx = 0;
    for (int i = 0; i < requestedDevicesCount; i++) {
      int deviceIdx = -1;
      if (cudaVisibleDevciesSetProperly)
        deviceIdx = i;
      else 
        deviceIdx = availableDevices[i];
      gpuErrchk( cudaSetDevice(deviceIdx), deviceIdx, errorMsgOut );

      //call some API functions to really occupy device (I'm lazy to look for more elegant way to do it)
      char * ddata;
      gpuErrchk( cudaMalloc(&ddata, 1), deviceIdx, errorMsgOut );
      gpuErrchk( cudaFree(ddata), deviceIdx, errorMsgOut );
      
      occupiedDevicesIdxs[nextDeviceIdx++] = deviceIdx;
    }
  } catch (const std::exception& e) {
    auto msg = string(e.what());
    memcpy(errorMsgOut, msg.c_str(), msg.length());
    return -1;
  }

  return 0;
}
