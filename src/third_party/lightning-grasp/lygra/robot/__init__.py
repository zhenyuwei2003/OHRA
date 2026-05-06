# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib

from lygra.robot.allegro import Allegro
from lygra.robot.shadow import Shadow
from lygra.robot.leap import Leap
from lygra.robot.dclaw import DClaw


def build_robot(name, urdf_path=None):
    if name == 'allegro':
        return Allegro(urdf_path=urdf_path)
    
    elif name == 'leap':
        return Leap(urdf_path=urdf_path)
    
    elif name == 'shadow':
        return Shadow(urdf_path=urdf_path)

    elif name == 'dclaw':
        return DClaw(urdf_path=urdf_path)

    elif name.startswith("leap_"):
        suffix = name.replace("leap_", "")
        module_name = f"lygra.robot.canonical.leap_hand_{suffix}"
        class_name = f"Leap{suffix}"
        module = importlib.import_module(module_name)
        Leap_CLS = getattr(module, class_name)
        return Leap_CLS(urdf_path=urdf_path)
    
    else:
        assert False, f"Robot {name} undefined."
