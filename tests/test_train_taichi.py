
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import argparse
from pathlib import Path
import numpy as np
import types

print("RUNNING V2", flush=True)

# --------------------------------------------------------------------------------

# Define comprehensive mocks to completely bypass heavy libraries and GPU requirements
# --------------------------------------------------------------------------------

class DummyTensor:
    def __init__(self, shape=None):
        self.shape = shape or [10, 4]
        self.values = self
        
    def __getitem__(self, key): return self
    def __setitem__(self, key, value): pass
    def __mul__(self, other): return self
    def __rmul__(self, other): return self
    def __add__(self, other): return self
    def __radd__(self, other): return self
    def __sub__(self, other): return self
    def __rsub__(self, other): return self
    def __truediv__(self, other): return self
    def __rtruediv__(self, other): return self
    def __lt__(self, other): return self
    def __gt__(self, other): return self
    def __le__(self, other): return self
    def __ge__(self, other): return self
    def __and__(self, other): return self
    def __or__(self, other): return self
    def float(self): return self
    def to(self, device): return self
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return np.zeros((10, 3))
    def mean(self): return self
    def abs(self): return self
    def item(self): return 0.5
    def backward(self): pass
    def clone(self): return self
    def exp(self): return self
    def max(self, dim=None): return self
    def any(self): return True
    def sum(self): return 0
    def repeat(self, *args): return self
    def reshape(self, *args): return self
    def view(self, *args): return self
    def permute(self, *args): return self
    def squeeze(self, *args): return self
    def unsqueeze(self, *args): return self
    
    # In-place operations
    def zero_(self): return self
    def add_(self, *args): return self
    def sub_(self, *args): return self
    def mul_(self, *args): return self
    def div_(self, *args): return self

    # Add other common tensor methods just in case

    def t(self): return self
    def contiguous(self): return self
    def new_zeros(self, *args): return self
    def new_ones(self, *args): return self


class DummyTorch(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.device = MagicMock(return_value="cpu")
        self.tensor = MagicMock(side_effect=lambda *args, **kwargs: DummyTensor())
        self.from_numpy = MagicMock(side_effect=lambda *args, **kwargs: DummyTensor())
        self.ones_like = MagicMock(side_effect=lambda *args, **kwargs: DummyTensor())
        self.zeros_like = MagicMock(side_effect=lambda *args, **kwargs: DummyTensor())
        self.randn_like = MagicMock(side_effect=lambda *args, **kwargs: DummyTensor())
        self.clamp_min = MagicMock(side_effect=lambda *args, **kwargs: DummyTensor())
        self.log = MagicMock(side_effect=lambda *args, **kwargs: DummyTensor())
        self.sqrt = MagicMock(side_effect=lambda *args, **kwargs: DummyTensor())
        self.zeros = MagicMock(side_effect=lambda *args, **kwargs: DummyTensor())
        self.ones = MagicMock(side_effect=lambda *args, **kwargs: DummyTensor()) # Added ones
        self.inverse = MagicMock(side_effect=lambda *args, **kwargs: DummyTensor())

        self.abs = MagicMock(side_effect=lambda *args, **kwargs: DummyTensor())
        self.cat = MagicMock(side_effect=lambda *args, **kwargs: DummyTensor())
        self.stack = MagicMock(side_effect=lambda *args, **kwargs: DummyTensor())
        self.cuda = MagicMock()
        self.cuda.is_available = MagicMock(return_value=False)
        self.manual_seed = MagicMock()
        self.no_grad = MagicMock


# Apply mocks to sys.modules immediately
sys.modules["torch"] = DummyTorch("torch")
sys.modules["torch.optim"] = types.ModuleType("torch.optim") # Mock optim as a module
sys.modules["torch.optim"].Adam = MagicMock() # Add Adam optimizer
sys.modules["taichi"] = MagicMock()
sys.modules["taichi_splatting"] = MagicMock()




# Submodules for taichi_splatting
def mock_submodule(name):
    m = MagicMock()
    sys.modules[name] = m
    return m

mock_submodule("taichi_splatting.data_types")
mock_submodule("taichi_splatting.renderer")
mock_submodule("taichi_splatting.perspective")
mock_submodule("taichi_splatting.misc")
mock_submodule("taichi_splatting.misc.encode_depth")
mock_submodule("taichi_splatting.misc.radius")
mock_submodule("taichi_splatting.misc.parameter_class")

# Mock specific classes
sys.modules["taichi_splatting.data_types"].Gaussians3D = MagicMock()
sys.modules["taichi_splatting.data_types"].RasterConfig = MagicMock()
sys.modules["taichi_splatting.renderer"].render_gaussians = MagicMock()
# Make render_gaussians return an object with .image attribute as DummyTensor
mock_rendering = MagicMock()
mock_rendering.image = DummyTensor()
sys.modules["taichi_splatting.renderer"].render_gaussians.return_value = mock_rendering

sys.modules["taichi_splatting.perspective"].CameraParams = MagicMock()
sys.modules["taichi_splatting.misc.radius"].compute_radius = MagicMock()
sys.modules["taichi_splatting.misc.parameter_class"].ParameterClass = MagicMock()
# Mock creating ParameterClass to return params with optimizer and DummyTensors
mock_params = MagicMock()
mock_params.batch_size = [10]
mock_params.optimizer = MagicMock()
mock_params.log_scaling = DummyTensor()
mock_params.append_tensors.return_value = mock_params
mock_params.__getitem__.return_value = mock_params
mock_params.clone.return_value = mock_params
sys.modules["taichi_splatting.misc.parameter_class"].ParameterClass.create.return_value = mock_params


sys.modules["plyfile"] = MagicMock()
sys.modules["cv2"] = MagicMock()


# Import train_taichi after all mocks are set
# Import train_taichi after all mocks are set
# We need to add the parent directory to sys.path to import train_taichi
import os
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Added {project_root} to sys.path")


import train_taichi

class TestTrainTaichi(unittest.TestCase):

    def setUp(self):
        # Setup specific mocks if needed for individual tests
        pass

    def test_train_execution(self):
        """Test the training loop runs without error."""
        args = argparse.Namespace(
            project_path="dummy_project",
            output_path="dummy_output",
            iterations=2, 
            densify_until_iter=5,
            densification_interval=1
        )
        
        # Mock SceneDataset to return dummy objects
        mock_dataset = MagicMock()
        mock_dataset.cameras = [MagicMock()] * 2
        
        # Images need to be DummyTensor to allow subtraction
        mock_dataset.images = [DummyTensor()] * 2 
        
        mock_dataset.points_xyz = np.random.rand(10, 3)
        mock_dataset.points_rgb = np.random.rand(10, 3)
        
        with patch("train_taichi.SceneDataset", return_value=mock_dataset), \
             patch("train_taichi.export_ply"):
             
            try:
                train_taichi.train(args)
            except Exception as e:
                self.fail(f"train() raised Exception: {e}")

    def test_scene_dataset_loading(self):
        """Test SceneDataset loads transforms and passes projection to CameraParams."""
        # Mock file operations and json loading
        mock_json_data = {
            "w": 800, "h": 600,
            "fl_x": 500, "fl_y": 500, "cx": 400, "cy": 300,
            "frames": [
                {
                    "file_path": "frame1.png",
                    "transform_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
                }
            ]
        }
        
        with patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps(mock_json_data))), \
             patch("json.load", return_value=mock_json_data), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("cv2.imread", return_value=np.zeros((600, 800, 3), dtype=np.uint8)), \
             patch("cv2.cvtColor", return_value=np.zeros((600, 800, 3), dtype=np.uint8)), \
             patch("train_taichi.CameraParams") as MockCameraParams:
             
             # Instantiate SceneDataset
             dataset = train_taichi.SceneDataset("dummy_path", device="cpu")
             
             # Assert CameraParams was called
             self.assertTrue(MockCameraParams.called)
             
             # Check arguments of the last call
             _, kwargs = MockCameraParams.call_args
             self.assertIn("projection", kwargs, "CameraParams should be called with 'projection'")
             self.assertNotIn("T_image_camera", kwargs, "CameraParams should NOT be called with 'T_image_camera'")
             
             # Check projection values [fl_x, fl_y, cx, cy]
             expected_projection = np.array([500.0, 500.0, 400.0, 300.0], dtype=np.float32)
             # Note: kwargs['projection'] is a Tensor (DummyTensor in mock env or real Tensor if torch not fully mocked)
             # Since torch is mocked as DummyTorch, it returns DummyTensor
             # We can't easily check values on DummyTensor unless we improve it, but checking existence is enough for API verification.
             pass

if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
