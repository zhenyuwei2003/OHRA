# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


class TorchGPUBufferPool:
    """ GPU Memory Pool based on torch tensor.
    """
    def __init__(self, device='cuda'):
        self.pool = {}
        self.device = device

    def request(self, name, shape, dtype=torch.float32):
        """ Request memory returned as a torch tensor. 
        """
        # Check if buffer already exists in the pool
        if name in self.pool:
            buffer, size, stored_dtype = self.pool[name]
            required_size = torch.prod(torch.tensor(shape)).item()
            if size >= required_size and stored_dtype == dtype:
                # Reuse
                return buffer[:required_size].view(*shape)

            else:
                # Allocate
                new_buffer = torch.empty(required_size, dtype=dtype, device=self.device)
                self.pool[name] = (new_buffer, required_size, dtype)
                return new_buffer.view(*shape)
        else:
            # If the buffer does not exist, allocate a new one
            required_size = torch.prod(torch.tensor(shape)).item()
            new_buffer = torch.empty(required_size, dtype=dtype, device=self.device)
            self.pool[name] = (new_buffer, required_size, dtype)
            return new_buffer.view(*shape)

    def release(self, name):
        if name in self.pool:
            del self.pool[name]

    def clear_pool(self):
        self.pool.clear()



class IKGPUBufferPool:
    """ Memory Buffer Pool Used by IK Solver.
        We allocate a large buffer in advance based on our usage estimate.
    """
    def __init__(self, n_dof, n_link, max_batch=32000, retry=10, max_contact=5):
        self.gpu_memory_pool = TorchGPUBufferPool()
        self.n_dof = n_dof 
        self.n_link = n_link

        self.gpu_memory_pool.request("ik_joint",         (max_batch * retry, n_dof, 4, 4), torch.float32)
        self.gpu_memory_pool.request("ik_link",          (max_batch * retry, n_link, 4, 4), torch.float32)
        self.gpu_memory_pool.request("ik_jac_result",    (n_dof, n_link * 6, max_batch * retry), torch.float32)
        self.gpu_memory_pool.request("ik_jac_error",     (max_batch, retry, max_contact, 6, n_dof), torch.float32)

    def get_ik_joint_buffer(self, batch, retry):
        return self.gpu_memory_pool.request("ik_joint", (batch * retry, self.n_dof, 4, 4))

    def get_ik_link_buffer(self, batch, retry):
        return self.gpu_memory_pool.request("ik_link", (batch * retry, self.n_link, 4, 4))

    def get_ik_jac_result_buffer(self, batch, retry):
        return self.gpu_memory_pool.request("ik_jac_result", (self.n_dof, self.n_link * 6, batch * retry))
    
    def get_ik_jac_error_buffer(self, batch, retry, n_contact):
        return self.gpu_memory_pool.request("ik_jac_error", (batch, retry, n_contact, 6, self.n_dof))

    # def get_ik_joint_buffer(self, shape):
    #     return self.gpu_memory_pool.request("ik_joint", shape)

    # def get_ik_link_buffer(self, shape):
    #     return self.gpu_memory_pool.request("ik_link", shape)

    # def get_ik_jac_result_buffer(self, shape):
    #     return self.gpu_memory_pool.request("ik_jac_result", shape)
    
    # def get_ik_jac_error_buffer(self, shape):
    #     return self.gpu_memory_pool.request("ik_jac_error", shape)
