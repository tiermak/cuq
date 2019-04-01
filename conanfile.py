from conans import ConanFile, CMake, tools

class CuqConan(ConanFile):
    name = "cuq"
    url = "https://github.com/PianeRamso/cuq"
    description = "CUDA multi-GPU concurrent tasks queue"
    settings = "os", "build_type", "arch"
    options = {"shared": [True, False]}
    default_options = "shared=False"
    generators = "cmake"
    exports_sources="*"

    def build_requirements(self):
        self.build_requires("boost/1.65.1@conan/stable")

    def build(self):
        cmake = CMake(self)
        self.run("cmake . ")
        self.run("make -j 8")

    def package(self):
        self.copy("*.so", dst="lib", keep_path=False)
        self.copy("*.h", dst="include", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["cuq"]
