import simpy

def picker(env):
    while True:
        print(f"{env.now: >4} s  Picking an item")
        yield env.timeout(6)
        print(f"{env.now: >4} s  Walking to next slot")
        yield env.timeout(10)

env = simpy.Environment()
env.process(picker(env))
env.run(until=60)

