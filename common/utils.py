

def make_env(args):
    if args.env_name == 'env_6':
        from env_6 import Environment
        # load scenario from script

        # create world
        # create multiagent environment
        env = Environment()
        args.n_agents = 2  # 需要操控的玩家个数，虽然敌人也可以控制，但是双方都学习的话需要不同的算法
        args.obs_shape = [53, 53]  # 每一维代表该agent的obs维度
        args.num_paths = [2, 2]  # 每一维代表该agent的act维度
        args.path_state_dim = [7,7]
        args.max_len = 3
        return env, args
    elif args.env_name == 'env_12':
        from env_12 import Environment
        env = Environment()
        args.n_agents = 4
        args.obs_shape = [209, 144, 144, 209]
        args.num_paths = [6, 4, 4, 6]
        args.path_state_dim = [7] *args.n_agents
        args.max_len = 4
        return env, args
    elif args.env_name == 'env_24':
        from env_24 import Environment
        env = Environment()
        args.n_agents = 6
        args.obs_shape = [282,317,317,317,317,282]
        args.num_paths = [6]*args.n_agents
        args.path_state_dim = [7] *args.n_agents
        args.max_len = 4
        return env,args