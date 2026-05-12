from pathlib import Path

path_dir_root = Path(__file__).parent.parent

path_data = path_dir_root.joinpath('dataset')
path_data.mkdir(exist_ok=True, parents=True)
path_train_data = path_data.joinpath('train')
path_train_data.mkdir(exist_ok=True, parents=True)
path_test_data = path_data.joinpath('test')
path_test_data.mkdir(exist_ok=True, parents=True)

path_artifacts = path_dir_root.joinpath('artifacts')
path_artifacts.mkdir(exist_ok=True, parents=True)

path_inference = path_dir_root.joinpath('inference')
path_inference.mkdir(exist_ok=True, parents=True)
path_images = path_inference.joinpath('images')
path_images.mkdir(exist_ok=True, parents=True)
path_results = path_inference.joinpath('results')
path_results.mkdir(exist_ok=True, parents=True)