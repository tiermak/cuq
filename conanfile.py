from conans import ConanFile

class Conan(ConanFile):
    name = "cuq"
    url = "https://github.com/PianeRamso/cuq"
    description = "CUDA multi-GPU concurrent tasks queue"
    settings = "os", "build_type", "arch","compiler","CUDA"
    options = {"shared": [True, False]}
    default_options = "shared=False"
    generators = "cmake"
    exports_sources="*"
    python_requires = "BcdConanRecipe/v1.0.0@biocad/biocad"
    python_requires_extend ="BcdConanRecipe.BcdConanRecipe"
