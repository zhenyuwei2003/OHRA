from pathlib import Path
import importlib.util
import torch 
import torch.nn.functional as F 
import time 

def load_gem_module():
    current_folder = Path(__file__).resolve().parent
    module_name = "geometry"
    module_path = current_folder / 'cpp' / 'build' / 'geometry' / 'geometry.so'

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_lbvh_module():
    current_folder = Path(__file__).resolve().parent
    module_name = "lbvh"
    module_path = current_folder / 'cpp' / 'build' / 'lbvh' / 'lbvh.so'

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    gem = load_gem_module()
except Exception as e:
    gem = None
    print("[Lygra Warning] Failed to load 'gem' module. Please make sure the required cuda binaries are set up and accessible.")
    print(f"  Error details: {e}")

try:
    lbvh = load_lbvh_module()
except Exception as e:
    lbvh = None
    print("[Lygra Warning] Failed to load 'lbvh' module. Please make sure the required cuda binaries are set up and accessible.")
    print(f"  Error details: {e}")