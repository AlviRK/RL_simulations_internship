import json
import pygame
from pathlib import Path
from affect.expression import Expression
from envs.room1 import Room1
LOG_DIR = Path(__file__).resolve().parent / "logs"


def replay_episodes(file=None, fps=5, express=False):
    if file is None:
        file = LOG_DIR / "replays.json"
    path = Path(file)
    if not path.is_absolute() and path.parent == Path("."):
        path = LOG_DIR / path


    data = json.loads(path.read_text())
    replays = data["replays"]

    env = Room1(variant=data["env_meta"]["variant"])

    pygame.init()
    screen = pygame.display.set_mode(((1+env.num_cols) * env.TILE_SIZE, (1+env.num_rows) * env.TILE_SIZE))
    pygame.display.set_caption("Room1 Environment")
    env.load_images() # Needs to be after screen init

    
    for replay in replays:
        episode_num = replay["episode"]
        actions = replay["actions"]
        insight = replay.get("insight") or None
        confidence = replay.get("confidence") or None
        curiosity = replay.get("curiosity") or None
        summary = replay.get("summary", {})
        total_reward = summary.get("total_reward", 0)
        title = f"Replay Episode {episode_num} - Total Reward: {total_reward}"
        pygame.display.set_caption(title)
        env.reset()
        if express:
            expression=Expression()
        
        for t, action in enumerate(actions):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            next_state, reward, done = env.step(action)

            
            env.render(screen)

            # pick per-step signal; fall back to previous/initial if length mismatch
            if express:
                if insight:
                    idx = min(t, len(insight) - 1)   # was t + 1
                    insight_signal = float(insight[idx])
                else:
                    insight_signal=0
                
                if confidence:
                    idx = min(t, len(confidence) - 1)   # was t + 1
                    confidence_signal = float(confidence[idx])
                else:
                    confidence_signal=0
                
                if curiosity:
                    idx = min(t, len(curiosity) - 1)   # was t + 1
                    curiosity_signal = float(curiosity[idx])
                else:
                    curiosity_signal=0
                
                #confidence = float(affect.simulate_confidence(agent.q_table[state]))
                expression.update(confidence_signal, curiosity_signal, insight_signal)
                expression.render(screen, env)
            

            #manage the default step delay and the screen clearing + buffer flipping
            pygame.time.delay(int(1000/fps))
            pygame.display.flip()
            screen.fill((0,0,0))

            if done:
                break

        
    pygame.quit()



def main():
    fps = 5
    LOG_DIR    = Path(__file__).resolve().parent / "replay" / "logs"
    OUT_FILE   = LOG_DIR / "replays.json"

    replay_episodes(file=OUT_FILE, fps=20, express=True)

if __name__ == "__main__":
    main()