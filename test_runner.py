import argparse
from env import Environment
from runner import Runner


args = argparse.Namespace(num_paths=3,buffer_size=10000)
env = Environment(args.num_paths)
runner = Runner(args,env)
runner.run()
