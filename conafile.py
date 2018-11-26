from conans import ConanFile, CMake, tools

class CuqConan(ConanFile):
    name = "cuq"
    url = "https://github.com/PianeRamso/cuq"
    description = "CUDA multi-GPU concurrent tasks queue"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    default_options = "shared=False"
    generators = "cmake"
    exports_sources="*"

    def build(self):
        self.run("pwd")
        cmake = CMake(self)
        self.run("cmake . ")
        self.run("make -j8")

    def package(self):
        self.copy("*.so", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["cuq"]

