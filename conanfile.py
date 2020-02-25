from conans import ConanFile

class Conan(ConanFile):
    python_requires = "BcdConanRecipe/v1.0.0@biocad/biocad"
    python_requires_extend ="BcdConanRecipe.BcdConanRecipe"
    name = "cuq"
    url = "https://github.com/biocad/cuq"
    description = "CUDA multi-GPU concurrent tasks queue"
    settings = "os", "build_type", "arch","compiler","CUDA"
    options = {"shared": [True, False]}
    default_options = "shared=True"
    generators = "cmake_find_package"
    exports_sources="*"

