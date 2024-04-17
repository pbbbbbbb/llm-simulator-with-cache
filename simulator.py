import numpy as np
import radix_tree
import time

class Simulator:
    def __init__(self, model=None) -> None:
        self.model = model
        self.cache = radix_tree.RadixTree()
        self.enc = self.cache.enc

    def run_simulations(self, prompt_list, enable_caching=True):
        times = []
        for prompt in prompt_list:
            start = time.time()
            self.run_single_sim(prompt, enable_caching=enable_caching)
            end = time.time()
            times.append(end - start)
        return times
        
    def run_single_sim(self, prompt, enable_caching):
        if not enable_caching:
            self.generate(prompt)
        else:    
            prompt = self.cache.enc.encode(prompt)
            l = len(prompt)
            kv = self.cache.get(prompt)
            if kv == None:
                self.generate(prompt)
                keys, vals = np.randint((l, l)), np.randint((l, l))
                self.cache.insert(prompt, keys, vals)
            else:
                keys, vals = kv
    
    def generate(self, prompt):
        if self.decode_time_per_token is None:
            raise Exception('Decode time is not set. Set the time with set_config.')
        prompt = self.cache.enc.encode(prompt)
        time.sleep(len(prompt) * self.decode_time_per_token)

    def set_config(
            self, 
            prefill_time_per_token, 
            decode_time_per_token
        ):
        self.prefill_time_per_token = prefill_time_per_token
        self.decode_time_per_token = decode_time_per_token