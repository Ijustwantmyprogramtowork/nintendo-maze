import time

"""
This script is a unit test. It also shows the syntax for instantiating Wallace.

Make sure that this script does not crash. Specifically, this should run in a limited time.

Note: this is not the evaluation script that we will use to evaluate your Wallace.
"""

LIMITED_TIME = 5*60

def test_wallace():
    from maze import create_maze
    from solution_mab_simple import BanditWallace as Wallace

    def run(env, wallace):
        n_played_steps = 1234
        obs = env.reset()
        gold = 0.
        done = False
        info = {}
        start = time.perf_counter()
        for _ in range(n_played_steps):
            if done:
                action = wallace.act(obs, gold, done)
                assert action is None
                obs = env.reset()
                gold = 0.
                done = False
            action = wallace.act(obs, gold, done)
            obs, gold, done, info = env.step(action, render_infos=wallace.get_custom_render_infos())
        end = time.perf_counter()
        if (end - start) > LIMITED_TIME:
            raise RuntimeError("This run took too much time! (%.2fsec)"%(end-start))
        golds = info['monitor.tot_golds']
        steps = info['monitor.tot_steps']
        print(f"Total golds gathered in {LIMITED_TIME/60} minutes: {golds} \nAnd Wallace walked {steps} steps")
        return golds

    mean_golds=0

    for exp_idx in range(5):
        env = create_maze(video_prefix="./video_%d"%exp_idx, overwrite_every_episode=False, fps=4)
        wallace = Wallace()
        mean_golds+=run(env, wallace)
        env.close()
    print("mean for 5 runs of golds=", mean_golds/5)

if __name__ == "__main__":
    print("BROAD EQUALS GREEN AND FOCUSED EQUALS VIOLET")
    test_wallace()
