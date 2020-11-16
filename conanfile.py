from conans import ConanFile, CMake

class Conan(ConanFile):
    name = "cuq"
    version = "0.12.0"
    url = "https://github.com/biocad/cuq"
    description = "CUDA multi-GPU concurrent tasks queue"
    settings = "os", "build_type", "arch", "compiler", "CUDA"
    generators = "cmake"
    exports_sources="*"

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        cmake.test()

    def package(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["cuq"]
        self.cpp_info.system_libs = ["cudart", "nvidia-ml", "stdc++"]

    def deploy(self):
        self.copy("*", dst="include", src="include")
        self.copy("*", dst="lib", src="lib")
