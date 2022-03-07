from common.arguments import get_args
from common.utils import make_env
from runner import Runner

if __name__ == "__main__":
    args = get_args()
    args.env_name = "env_6"
    env,args = make_env(args)
    runner = Runner(args,env)
    runner.run()

