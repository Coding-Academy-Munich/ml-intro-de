# %%
import logging
import platform
from multiprocessing import cpu_count
from pathlib import Path

from pydantic_settings import BaseSettings
from appdirs import user_data_dir, user_cache_dir

# %%
COURSES_SUBDIR = "coding_academy/python_courses/"


# %%
class EnvConfig(BaseSettings):
    # Paths
    base_dir_path: Path = Path(user_data_dir()) / COURSES_SUBDIR
    data_dir_path: Path = base_dir_path / "data/"
    models_dir_path: Path = data_dir_path / "models/"
    cache_dir_path: Path = Path(user_cache_dir()) / COURSES_SUBDIR
    sklearn_cache_dir_path: Path = cache_dir_path / "sklearn/"
    pickle_dir_path: Path = cache_dir_path / "pickles/"

    # Individual files
    mnist_pkl_path: Path = pickle_dir_path / "mnist.pkl"
    fashion_mnist_pkl_path: Path = pickle_dir_path / "fashion-mnist.pkl"
    processed_mnist_pkl_path: Path = pickle_dir_path / "mnist-processed.pkl"
    processed_fashion_mnist_pkl_path: Path = (
        pickle_dir_path / "fashion-mnist-processed.pkl"
    )
    california_housing_pkl_path: Path = pickle_dir_path / "california-housing.pkl"
    processed_california_housing_pkl_path: Path = (
        pickle_dir_path / "california-housing-processed.pkl"
    )

    # Logging
    logfile: Path = base_dir_path / "default.log"
    loglevel: int = logging.DEBUG

    # Multiprocessing
    max_processes: int = 32 if platform.system() == "Windows" else cpu_count()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for dir_path in [
            self.base_dir_path,
            self.data_dir_path,
            self.cache_dir_path,
            self.models_dir_path,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=self.loglevel, filename=self.logfile)

    class Config:
        env_prefix = "cam_python_courses_"
