from conans import ConanFile, CMake, tools, python_requires

bcd_conan_recipe = python_requires("BcdConanRecipe/v1.0.0@biocad/biocad")

class Conan(bcd_conan_recipe.BcdConanRecipe):
    name = "cuq"
    url = "https://github.com/PianeRamso/cuq"
    description = "CUDA multi-GPU concurrent tasks queue"
    settings = "os", "build_type", "arch","compiler","CUDA"
    options = {"shared": [True, False]}
    default_options = "shared=False"
    generators = "cmake"
    exports_sources="*"
    requires = "cummon/v0.3.0@biocad/biocad"
