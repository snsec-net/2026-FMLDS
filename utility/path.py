from pathlib import Path

path_dir_root = Path(__file__).parent.parent

path_data = path_dir_root.joinpath('dataset')
path_data.mkdir(exist_ok=True, parents=True)

path_artifacts = path_dir_root.joinpath('artifacts')
path_artifacts.mkdir(exist_ok=True, parents=True)

# path_figure = path_dir_root.joinpath(f'figure')
# path_figure.mkdir(exist_ok=True, parents=True)