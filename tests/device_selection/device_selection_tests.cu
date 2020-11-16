#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <stdlib.h>
#include <iostream>
#include <string>

#include <cuq.h>

#include <cuda_runtime.h>

using namespace std;

int devices[128];
char errorMsg[1000];

TEST_CASE("GPU occupation", "[occupation]") {
  CHECK(occupyDevices(1, devices, errorMsg) == 0);
  CHECK(devices[0] == 0);
}

TEST_CASE("GPU occupation with CUDA_VISIBLE_DEVICES=0", "[occupation][CUDA_VISIBLE_DEVICES]") {
  putenv((char*)"CUDA_VISIBLE_DEVICES=0");

  CHECK(occupyDevices(1, devices, errorMsg) == 0);
  CHECK(devices[0] == 0);

  putenv((char*)"CUDA_VISIBLE_DEVICES=");
}

TEST_CASE("GPU occupation with CUDA_VISIBLE_DEVICES=\"0\"", "[occupation][CUDA_VISIBLE_DEVICES]") {
  putenv((char*)"CUDA_VISIBLE_DEVICES=\"0\"");

  CHECK(occupyDevices(1, devices, errorMsg) == 0);
  CHECK(devices[0] == 0);

  putenv((char*)"CUDA_VISIBLE_DEVICES=");
}

TEST_CASE("GPU occupation with CUDA_VISIBLE_DEVICES='0'", "[occupation][CUDA_VISIBLE_DEVICES]") {
  putenv((char*)"CUDA_VISIBLE_DEVICES='0'");

  CHECK(occupyDevices(1, devices, errorMsg) == 0);
  CHECK(devices[0] == 0);

  putenv((char*)"CUDA_VISIBLE_DEVICES=");
}

TEST_CASE("GPU occupation with CUDA_VISIBLE_DEVICES=0,1", "[occupation][CUDA_VISIBLE_DEVICES]") {
  putenv((char*)"CUDA_VISIBLE_DEVICES=0,1");

  CHECK(occupyDevices(1, devices, errorMsg) == 0);
  CHECK(devices[0] == 0);

  putenv((char*)"CUDA_VISIBLE_DEVICES=");
}

TEST_CASE("A process tries to occupy the same GPU two times", "[occupation]") {
  CHECK(occupyDevices(1, devices, errorMsg) == 0);
  CHECK(devices[0] == 0);

  //Do it once again to be sure it works if a process which already occupied a GPU requests to occupy it one more time
  CHECK(occupyDevices(1, devices, errorMsg) == 0);
  CHECK(devices[0] == 0);
}

TEST_CASE("More GPUs than in CUDA_VISIBLE_DEVICES are requested", "[occupation][CUDA_VISIBLE_DEVICES]") {
  putenv((char*)"CUDA_VISIBLE_DEVICES=0");

  CHECK(occupyDevices(2, devices, errorMsg) == -1);

  putenv((char*)"CUDA_VISIBLE_DEVICES=");
}

TEST_CASE("More GPUs than exists on the machine are requested", "[occupation]") {
  int realDeviceCount;
  cudaGetDeviceCount(&realDeviceCount);

  CHECK(occupyDevices(realDeviceCount + 1, devices, errorMsg) == -1);
}
