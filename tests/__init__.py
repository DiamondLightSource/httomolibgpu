import cupy as cp

class MaxMemoryHook(cp.cuda.MemoryHook):
    
    def __init__(self, initial=0):
        self.max_mem = initial
        self.current = initial
    
    def malloc_postprocess(self, device_id: int, size: int, mem_size: int, mem_ptr: int, pmem_id: int):
        self.current += mem_size
        self.max_mem = max(self.max_mem, self.current)

    def free_postprocess(self, device_id: int, mem_size: int, mem_ptr: int, pmem_id: int):
        self.current -= mem_size

    def alloc_preprocess(self, **kwargs):
        pass

    def alloc_postprocess(self, device_id: int, mem_size: int, mem_ptr: int):
        pass
    
    def free_preprocess(self, **kwargs):
        pass

    def malloc_preprocess(self, **kwargs):
        pass
