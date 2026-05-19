import pygame
import json
import random
import numpy as np
from pathlib import Path

from envs.room1 import Room1
from agents.td_agent import TDAgent
from agents.sarsa_agent import SarsaAgent
from replay.replay import replay_episodes
from affect.insight import Insight
from affect.confidence import Confidence
from affect.curiosity import Curiosity
from affect.expression import Expression



def train_and_save(
    *,
    episodes, 
    max_steps,
    save_every,
    seed,
    out_file,
    render=False,
    fps=5,
    signals=False,
    variant="1",
    agent_class=TDAgent,
    #custom_settings=None,
):
    # --- 1. Initialize Env and Agent with settings ---
    # make env
    env = Room1(variant=variant)
    
    
    # agent settings
    settings = {
        "alpha": 0.1,      # Learning rate 0.1
        "gamma": 0.9,      # Discount factor
        "epsilon": 0.1,    # Exploration rate
        "epsilon_decay": 4.75e-05,  # Decay rate for exploration, before: 2.5e-05
        "epsilon_min": 0.01,  # Minimum exploration rate; alternative 0.01
        "seed": seed
    }

    # if custom_settings:
    #     settings.update(custom_settings)

    agent = agent_class(env, settings)

    # --- 2. Define Master Dict for data logging ---

    data = {
        "env_meta": {                                           # Environment meta data
            "env": env.__class__.__name__,                      # Which env class
            "variant": str(variant),                            # Which layout
            "num_rows": int(env.num_rows),
            "num_cols": int(env.num_cols),
            "num_cells": int(env.num_cells),
            "start_cell": int(env.start_cell),
            "terminal_cell": int(env.terminal_cell),
            "dirt_cells": [int(x) for x in env.dirt_cells],     # Which ones are dirt cells
            "vase_cells": [int(x) for x in env.vase_cells],     # Which ones are vase cells
        },

        "agent": {
            "name": agent_class.__name__,                       # Which agent was used
            "settings": {                                       # Agent settings
                "alpha": float(settings["alpha"]),
                "gamma": float(settings["gamma"]),
                "epsilon_start": float(settings["epsilon"]),
                "epsilon_decay": float(settings["epsilon_decay"]),
                "epsilon_min": float(settings["epsilon_min"]),
                "seed": int(settings["seed"]),
            }
        },

        "train_meta": {                                         # Training meta data
            "episodes_total": int(episodes),
            "max_steps": int(max_steps),
            "save_every": int(save_every) if save_every else None,
            "rendered": bool(render),
            "simulated_signals": bool(signals),
            "fps": int(fps)
        },

        "replays": []                                      # Logging data
    }

    # --- Rendering & Replay Dynamics ---

    if render:
        pygame.init()
        screen = pygame.display.set_mode(((1+env.num_cols) * env.TILE_SIZE, (1+env.num_rows) * env.TILE_SIZE))
        pygame.display.set_caption("Room1 Environment")
        env.load_images() # Needs to be after screen init
        
    confidence = Confidence() if signals else None # because i dont want to reset the count

    for episode in range(episodes):
        state = env.reset()
        if signals:
             #create an signal simulators and expression object
            insight=Insight(window=5)
            curiosity = Curiosity()
            expression=Expression()

            confidence.reset()

        done = False
        total_reward = 0
        steps = 0

        replay_record = ((episode == 0) or (save_every and ((episode + 1) % save_every == 0)))
        #step-data
        actions_buffer = []
        policy_change_buffer = []
        insight_buffer = []
        confidence_buffer = []
        curiosity_buffer = []

        states_buffer = []
        next_states_buffer = []
        rewards_buffer = []
        td_error_buffer = []
        events_buffer = []  

        #state counts        
        state_visits = {} 

        #"firsts" tracking
        num_dirts_found = 0
        num_vases_hit = 0

        first_dirt_step = None
        second_dirt_step = None
        first_vase_step = None
        exit_step = None
        revisit_reward_steps = []

        #already rewarded cells
        rewarded_cells = set()

        while not done and steps < max_steps:

               
            #Do agent execution and learning

            ## 1. Policy change log 1: Check greedy action before learning (LOGGING ONLY)
            q_values_before = agent.q_table[state] # Q values before learning
            greedy_action_before = np.argmax(q_values_before)

            ## 2. Execution of Action Selection + Update/Learning
            action, exploratory = agent.action_selection(state)
            next_state, reward, done = env.step(action)
            td_error, td_target = agent.update_rule(state, action, reward, next_state, done)
            
            ## 3. Policy change log 2: Check greedy action after learning (LOGGING ONLY)
            q_values_after = agent.q_table[state] # Q-values after learning
            greedy_action_after = np.argmax(q_values_after)


            ## 4. Policy change log (LOGGING ONLY)
            policy_changed = int(greedy_action_before != greedy_action_after) # 1 if yes

            #Only do signal simulation ad expression management when we have "emotions" (affect=True)
            if signals:
                insight_val = float(insight.update(td_error))
                confidence_val = float(confidence.update(state, action, env.n_actions()))
                curiosity_val = float(curiosity.update(agent.q_table[state]))
                expression.update(confidence_val,curiosity_val,insight_val)
            else:
                insight_val = 0
                confidence_val = 0
                curiosity_val = 0

            # Add this inside the training loop after insight_val calculation
            if signals and reward == 0.0 and insight_val > 0.7:
                print(f"--- HIGH FLOOR INSIGHT DETECTED ---")
                print(f"Episode: {episode+1}, Step: {steps}")
                print(f"State: {state} -> Next State: {next_state}")
                print(f"TD Error: {td_error:.4f}, Insight: {insight_val:.4f}")
                print(f"V(s) was: {agent.q_table[state][action]:.4f}")
                print(f"V(s') max is: {np.max(agent.q_table[next_state]):.4f}")
                        
            #only do all the following when we want to see it (render=True)
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    #adjust the FPS with key up or down
                    keys=pygame.key.get_pressed()

                    if keys[pygame.K_UP]:
                        fps=fps+1
                    if keys[pygame.K_DOWN]:
                        fps=min(fps-1, 1)
                
                #render the emv and potentially the afective state expression
                env.render(screen)
                if signals:
                    expression.render(screen, env)

                #manage the default step delay and the screen clearing + buffer flipping
                pygame.time.delay(int(1000/fps))
                pygame.display.flip()
                screen.fill((0,0,0))

            if replay_record:
                step_events = []

                # core step data
                actions_buffer.append(int(action))
                states_buffer.append(int(state))               
                next_states_buffer.append(int(next_state))  
                rewards_buffer.append(float(reward))
                td_error_buffer.append(float(td_error))
                policy_change_buffer.append(policy_changed) # For Analysis Metrics (LOGGING ONLY)

                # signals
                insight_buffer.append(float(insight_val))
                confidence_buffer.append(float(confidence_val))
                curiosity_buffer.append(float(curiosity_val))

                # visitation (by STATE)
                state_visits[state] = state_visits.get(state, 0) + 1

                # event detection
                num_cells = env.num_cells
                bits      = int(state // num_cells)
                next_bits = int(next_state // num_cells)
                next_cell = int(next_state % num_cells)

                # Dirt discovery
                if (next_bits != bits) and (reward == 0.5):
                    num_dirts_found += 1
                    rewarded_cells.add(next_cell)
                    if first_dirt_step is None:
                        first_dirt_step = steps
                        step_events.append("FIRST_DIRT")
                    elif (second_dirt_step is None) and (num_dirts_found == 2):
                        second_dirt_step = steps
                        step_events.append("SECOND_DIRT")

                # Vase hit
                if reward == -1.0:
                    if first_vase_step is None:
                        first_vase_step = steps
                        step_events.append("FIRST_VASE")
                    num_vases_hit += 1

                # Exit
                if (reward == 1.0) and (exit_step is None):
                    exit_step = steps
                    step_events.append("EXIT")

                # Revisit previously rewarding tile (now 0 reward)
                if (reward == 0.0) and (next_cell in rewarded_cells):
                    revisit_reward_steps.append(steps)
                    step_events.append("REVISIT_REWARD_TILE")

                events_buffer.append(step_events)



            state = next_state
            total_reward += reward
            steps += 1
                            
        if replay_record:
            replay_entry = {
                "episode": episode + 1,                                 # Episode number

                # step-aligned arrays
                "actions": actions_buffer,                              # Actions within an episode
                "states": states_buffer,                                # Visited states within an episode
                "next_states": next_states_buffer,                      # Next States within an episode
                "rewards": rewards_buffer,                              # Rewards within an episode
                "td_error": td_error_buffer,                            # TD Errors within an episode
                "policy_changes": policy_change_buffer,                 # Policy changes within an episode
                "events": events_buffer,                                # Events within an episode

                # signals
                "insight": insight_buffer,                              # Insight signals within an episode
                "confidence": confidence_buffer,                        # Confidence signals within an episode
                "curiosity": curiosity_buffer,                          # Curiosity signals within an episode

                # summaries
                "summary": {
                    "first_dirt_step": first_dirt_step,
                    "second_dirt_step": second_dirt_step,
                    "first_vase_step": first_vase_step,
                    "exit_step": exit_step,
                    "revisit_reward_steps": revisit_reward_steps,
                    "num_dirts_found": num_dirts_found,
                    "num_vases_hit": num_vases_hit,
                    "steps": steps,
                    "total_reward": float(total_reward),
                },

                # per-episode visitation (by STATE)
                "state_visits": {str(k): int(v) for k, v in state_visits.items()},
            }
            data["replays"].append(replay_entry)



        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Steps: {steps}")


    

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2)) 
    print(f"\nSaved {len(data['replays'])} replays to {out_path.resolve()}")

    if render:
        pygame.quit()
    
    return data



def main():
    # Define a list of different random seeds to make your paper scientifically rigorous!
    #SEEDS = [42, 43, 44, 45, 46]
    SEEDS = [42] 
    VARIANT = "1"
    
    # Get the directory where we want to save logs
    log_dir = Path(__file__).resolve().parent / "replay" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        print(f"\n=========================================")
        print(f" STARTING TRAINING FOR SEED: {seed}")
        print(f"=========================================")
        
        # We give each file a unique name based on its seed
        out_file = log_dir / f"seed_{seed}.json"

        train_and_save(
            episodes=300,   # was 100
            max_steps=100,
            save_every=1,
            seed=seed,
            out_file=out_file,
            variant=VARIANT,
            signals=True, 
            render=True,
            fps=10
        )
        
    print("\n All seeds have finished training!")

if __name__ == "__main__":
    main()