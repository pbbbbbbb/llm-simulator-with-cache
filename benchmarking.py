import matplotlib.pyplot as plt
import simulator

def test_prompt_length():
    length = [2 ** x for x in range(13)]
    prompt = 'vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.'
    with_cache = []
    without_cache = []
    N = 100
    for n in length:
        sim = simulator.Simulator()
        sim.set_config(prefill_time_per_token=0.25*0.001, decode_time_per_token=0.25*0.001)
        # prompt = sim.cache.enc.encode(PROMPT)
        prompts = [(prompt*(n//len(prompt)))[:n]] * N

        t = sim.run_simulations(prompts, enable_caching=False)
        without_cache.append(sum(t) / len(t))
        print(without_cache[-1])

        t = sim.run_simulations(prompts, enable_caching=True)
        with_cache.append(sum(t) / len(t))
        print(with_cache[-1])

    plt.plot(range(13), without_cache)
    plt.plot(range(13), with_cache)
    plt.xticks(ticks=range(13), labels=[str(l) for l in length])
    plt.xlabel('sequence length')
    plt.ylabel('time (ms)')
    plt.legend(['without cache', 'with cache'])
    plt.savefig('prompt_length')
        
if __name__ == "__main__":
    test_prompt_length()
