import argparse
from env import Environment
from runner import Runner


args = argparse.Namespace(num_paths=3,buffer_size=10000,batch_size=3,num_nodes=7,num_edges=12,gamma=0.95)
env = Environment(args.num_paths)
runner = Runner(args,env)
runner.run()

