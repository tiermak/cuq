#include <iostream>
#include <set>
#include <string>
#include <sstream>
#include <unistd.h>

#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

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
using namespace boost::interprocess;

set<int> getAllPhysicallyAvailableDevices() {

  unsigned int i, n;
  NVML( nvmlInit() );
  NVML( nvmlDeviceGetCount(&n) );
  if (n > 128) {
    string msg = "nvmlDeviceGetCount returned " +  to_string(n) + ". This does not make sense.";
    throw std::runtime_error(msg);
  }

  set<int> res;

  int currentPid = getpid();

  for (i = 0; i < n; i++) {
    nvmlDevice_t dev;
    unsigned int num_procs = 128;
    nvmlProcessInfo_t procs[128];
    NVML( nvmlDeviceGetHandleByIndex(i, &dev) );
    NVML( nvmlDeviceGetComputeRunningProcesses(dev, &num_procs, procs) );
    if (num_procs < 1) {
      res.insert(i);
      // cout << "getAllPhysicallyAvailableDevices:, i: " << i << endl;
    } else {
      //also add deviceIdx if current process occupied this device already
      for (int p = 0; p < num_procs; p++){
        if (procs[p].pid == currentPid)
          res.insert(i);
      }
    }
  }

  return res;
}

set<int> readCudaVisibleDevices() {
  set<int> res;

  auto rawValue = std::getenv("CUDA_VISIBLE_DEVICES");

  if (!rawValue)
    return res;

  string visibleDevices = string(rawValue);
  stringstream ss(visibleDevices);

  if (ss.peek() == '"' || ss.peek() == '\'')
    ss.ignore();

  int i;
  while (ss >> i){
    res.insert(i);

    // cout << "readCudaVisibleDevices:, i: " << i << endl;

    if (ss.peek() == ',' || ss.peek() == '"' || ss.peek() == '\'')
      ss.ignore();
  }

  return res;
}

set<int> getAvailableDevices() {
  set<int> visibleDevices = readCudaVisibleDevices();

  if (visibleDevices.empty())
    return getAllPhysicallyAvailableDevices();

  return visibleDevices;
}

extern "C"
int occupyDevices(int requestedDevicesCount, int * occupiedDevicesIdxs, char * errorMsg) {
  named_mutex mutex(open_or_create, "process_safe_device_selection_mutex");
  
  scoped_lock<named_mutex> lock(mutex, try_to_lock);

  try {
    set<int> availableDevcices = getAvailableDevices();

    if ((int)availableDevcices.size() < requestedDevicesCount) {
      string msg = "There are not as many free devices as requested. Requested devices count: " 
        + to_string(requestedDevicesCount) + ". Available devices count: " + to_string(availableDevcices.size()) + ".";
      
      memcpy(errorMsg, msg.c_str(), msg.length());

      return -1;
    }

    int nextDeviceIdx = 0;
    for (auto it = availableDevcices.begin(); it != availableDevcices.end(); ++it) {
      int deviceIdx = *it;
      gpuErrchk( cudaSetDevice(deviceIdx), deviceIdx, errorMsg );

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

// #include <chrono>
// #include <thread>

// int main() {
//   cout << "startred..." << endl;

//   int devicesIdxs[32];
//   char msg[1000] = "";
  
//   int res = occupyDevices(1, devicesIdxs, msg);

//   cout << "res: " << res << " msg: " << string(msg) << endl;

//   cout << "sleep..." << endl;
//   std::this_thread::sleep_for(std::chrono::milliseconds(10000));
  
//   res = occupyDevices(1, devicesIdxs, msg);

//   cout << "res: " << res << " msg: " << string(msg) << endl;

//   cout << "sleep..." << endl;
//   std::this_thread::sleep_for(std::chrono::milliseconds(10000));

//   cout << "finished!" << endl;

//   return 0;
// }

