import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import os
import importlib.util

class Test3DScan(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Import the module safely now that it is guarded
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Mocks for dependencies
        sys.modules['nerfstudio'] = MagicMock()
        sys.modules['nerfstudio.process_data.colmap_utils'] = MagicMock()
        sys.modules['plyfile'] = MagicMock()
        sys.modules['torch'] = MagicMock()
        
        # Load module
        spec = importlib.util.spec_from_file_location("scan_module", "3d-scan.py")
        cls.scan_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.scan_module)

    def test_find_input_video_logic(self):
        """Test the standalone find_input_video logic"""
        # Patch the Path GLOBALLY imported in the module, not pathlib.Path
        with patch.object(self.scan_module, 'Path') as MockPath:
            # We also need to make sure the mocked instances behave correctly
            # Path("/kaggle/input") -> mock_kaggle
            # Path("input") -> mock_local
            
            mock_kaggle = MagicMock()
            mock_local = MagicMock()
            
            def path_side_effect(arg):
                if arg == "/kaggle/input":
                    return mock_kaggle
                if arg == "input":
                    return mock_local
                return MagicMock()
            
            MockPath.side_effect = path_side_effect
            
            # Scenario 1: Kaggle has video
            mock_kaggle.exists.return_value = True
            mock_kaggle.rglob.return_value = ["found_kaggle.mp4"]
            
            res = self.scan_module.find_input_video()
            self.assertEqual(res, "found_kaggle.mp4")
            
            # Scenario 2: Kaggle empty, Local has video
            # Reset checks
            mock_kaggle.exists.return_value = True # dir exists
            mock_kaggle.rglob.return_value = []    # but empty
            
            mock_local.exists.return_value = True
            mock_local.rglob.return_value = ["found_local.mp4"]
            
            res = self.scan_module.find_input_video()
            self.assertEqual(res, "found_local.mp4")
            
    @patch('subprocess.run')
    def test_run_command(self, mock_subprocess):
        """Test command wrapper"""
        self.scan_module.run_command("echo test", shell=True)
        mock_subprocess.assert_called_with("echo test", shell=True, check=True)

if __name__ == '__main__':
    unittest.main()
