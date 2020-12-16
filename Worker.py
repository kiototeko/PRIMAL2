
import scipy.signal as signal
import copy
import numpy as np
import ray
import os
import imageio
import time
from od_mstar3.col_set_addition import OutOfTimeError, NoSolutionError
from od_mstar3 import od_mstar
from od_mstar3 import cpp_mstar
from GroupLock import Lock
from operator import sub, add
from warehouse_env import Action
import matplotlib.pyplot as plt
import matplotlib.animation as animation




from parameters import *

# helper functions
def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Worker():
    def __init__(self, metaAgentID, workerID, workers_per_metaAgent, env, localNetwork, sess, groupLock, learningAgent, global_step):
        
        self.metaAgentID = metaAgentID
        self.agentID = workerID
        self.name = "worker_" + str(workerID)
        self.num_workers = workers_per_metaAgent
        self.global_step = global_step
        self.nextGIF = 0
        
        self.env = env
        self.local_AC = localNetwork
        self.groupLock = groupLock
        self.learningAgent = learningAgent
        self.sess = sess
        self.allGradients = []

    def calculateImitationGradient(self, rollout, episode_count):
        rollout = np.array(rollout, dtype=object)
        # we calculate the loss differently for imitation
        # if imitation=True the rollout is assumed to have different dimensions:
        # [o[0],o[1],optimal_actions]
        
        temp_actions = np.stack(rollout[:,2])
        rnn_state = self.local_AC.state_init
        feed_dict = {self.global_step                  : episode_count,
                     self.local_AC.inputs         : np.stack(rollout[:, 0]),
                     self.local_AC.goal_pos       : np.stack(rollout[:, 1]),                     
                     self.local_AC.optimal_actions: np.stack(rollout[:, 2]),
                     self.local_AC.state_in[0]    : rnn_state[0],
                     self.local_AC.state_in[1]    : rnn_state[1],
                     self.local_AC.train_imitation : (rollout[:, 3]),
                     self.local_AC.target_v   : np.stack(temp_actions),
                     self.local_AC.train_value: temp_actions,

                     }


        v_l, i_l, i_grads = self.sess.run([self.local_AC.value_loss,
                                           self.local_AC.imitation_loss,
                                           self.local_AC.i_grads],
                                          feed_dict=feed_dict)


        return [i_l], i_grads
    
    def calculateGradient(self, rollout, bootstrap_value, episode_count, rnn_state0):
        # ([s,a,r,s1,v[0,0]])

        rollout = np.array(rollout, dtype=object)
        observations = rollout[:, 0]
        goals = rollout[:, -3]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 4]
        valids = rollout[:, 5]
        train_value = rollout[:, -2]
        train_policy = rollout[:,-1] 

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        num_samples = min(EPISODE_SAMPLES, len(advantages))
        sampleInd = np.sort(np.random.choice(advantages.shape[0], size=(num_samples,), replace=False))

        feed_dict = {
            self.global_step              : episode_count,
            self.local_AC.target_v   : np.stack(discounted_rewards),
                self.local_AC.inputs     : np.stack(observations),
                self.local_AC.goal_pos   : np.stack(goals),
                self.local_AC.actions    : actions,
                self.local_AC.train_valid: np.stack(valids),
                self.local_AC.advantages : advantages,
                self.local_AC.train_value: train_value,
                self.local_AC.state_in[0]: rnn_state0[0],
                self.local_AC.state_in[1]: rnn_state0[1],
                self.local_AC.train_policy: train_policy,
                self.local_AC.train_valids : np.vstack(train_policy) 
            }

        v_l, p_l, valid_l, e_l, g_n, v_n, grads = self.sess.run([self.local_AC.value_loss,
                                                            self.local_AC.policy_loss,
                                                            self.local_AC.valid_loss,
                                                            self.local_AC.entropy,
                                                            self.local_AC.grad_norms,
                                                            self.local_AC.var_norms,
                                                            self.local_AC.grads],
                                                           feed_dict=feed_dict)

        return [v_l, p_l, valid_l, e_l, g_n, v_n], grads



    def imitation_learning_only(self, episode_count):      
        #print("start imitation")
        self.env.reset()
        for i in range(1, self.num_workers+1):
            new_goal_location = self.env.get_new_goal_location()
            self.env.assign_goal(i-1,new_goal_location)
            

        rollouts, targets_done = self.parse_path(episode_count)

        if rollouts is None:
            return None, 0

        gradients = []
        losses = []
        for i in range(self.num_workers):
            train_buffer = rollouts[i]
            
            imitation_loss, grads = self.calculateImitationGradient(train_buffer, episode_count)

            gradients.append(grads)
            losses.append(imitation_loss)

        #print("done imitation")
        return gradients, losses

    
    
    def run_episode_multithreaded(self, episode_count, coord):
        
        if self.metaAgentID < NUM_IL_META_AGENTS:
            assert(1==0)
            #print("THIS CODE SHOULD NOT TRIGGER")
            self.is_imitation = True
            self.imitation_learning_only()


        global episode_lengths, episode_mean_values, episode_invalid_ops, episode_stop_ops, episode_rewards, episode_finishes
        

        num_agents = self.num_workers

        
        with self.sess.as_default(), self.sess.graph.as_default():
            while self.shouldRun(coord, episode_count):
                episode_buffer, episode_values = [], []
                episode_reward = episode_step_count = episode_inv_count = targets_done =episode_stop_count = 0
            
                # Initial state from the environment
                if self.agentID == 1:
                    self.env.reset()
                    for i in range(1, self.num_workers+1):
                        self.env.assign_goal(i-1,self.env.get_new_goal_location())
                        joint_observations[self.metaAgentID][i] = self.env._observe(i-1)
         
                    

                self.synchronize()  # synchronize starting time of the threads

                # Get Information For Each Agent 
                validActions = self.env.listValidActions(self.agentID-1) #Should we just ignore validActions?
                
                s = joint_observations[self.metaAgentID][self.agentID]
                s = s[1:5] #We only use 4 of the channels
                #s[0] = self.env.obstacle_map #Substitute agent map for obstacle map
                goal_vector = self.goal_vector_calc(self.agentID)
                
                

                rnn_state = self.local_AC.state_init
                rnn_state0 = rnn_state

                self.synchronize()  # synchronize starting time of the threads
                swarm_reward[self.metaAgentID] = 0
                swarm_targets[self.metaAgentID] = 0

                episode_rewards[self.metaAgentID] = []
                episode_finishes[self.metaAgentID] = []
                episode_lengths[self.metaAgentID] = []
                episode_mean_values[self.metaAgentID] = []
                episode_invalid_ops[self.metaAgentID] = []
                episode_stop_ops[self.metaAgentID] = []


                # ===============================start training =======================================================================
                # RL
                if True:
                    # prepare to save GIF
                    saveGIF = False
                    global GIFS_FREQUENCY_RL
                    if OUTPUT_GIFS and self.agentID == 1 and ((not TRAINING) or (episode_count >= self.nextGIF)):
                        saveGIF = True
                        self.nextGIF = episode_count + GIFS_FREQUENCY_RL
                        GIF_episode = int(episode_count)
                        GIF_frames = [self.env.render(zoom_size=30, agent_id=None)]

                    # start RL
                    finished = False
                    while not finished:
                        a_dist, v, rnn_state = self.sess.run([self.local_AC.policy,
                                                         self.local_AC.value,
                                                         self.local_AC.state_out],
                                                        feed_dict={self.local_AC.inputs     : [s],  # state doesn't inclue agent map
                                                                   self.local_AC.goal_pos   : [goal_vector],  # goal vector
                                                                   self.local_AC.state_in[0]: rnn_state[0],
                                                                   self.local_AC.state_in[1]: rnn_state[1]})

                        skipping_state = False 
                        train_policy = train_val = 1 
                       
                        if not skipping_state :
                            if not (np.argmax(a_dist.flatten()) in validActions):
                                episode_inv_count += 1
                                train_val = 0 
                            train_valid = np.zeros(a_size)
                            train_valid[validActions] = 1

                            valid_dist = np.array([a_dist[0, validActions]])
                            valid_dist /= np.sum(valid_dist)

                            a = validActions[np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]
                            joint_actions[self.metaAgentID][self.agentID] = a
                            if a == 0 :
                                episode_stop_count += 1

                        # Make A Single Agent Gather All Information

                        self.synchronize()
                        
                        #print("current State: ", self.agentID, " state: ", self.env.agent_state[self.agentID-1])
                        #print("agent: ", self.agentID, " action: ", Action(joint_actions[self.metaAgentID][self.agentID]))

                        if self.agentID == 1:
                            for i in range(1, self.num_workers+1):
                                #print("Before agent: ", i-1, " goal: ", self.env.agent_goal[i-1], " state: ", self.env.agent_state[i-1], " action: ", Action(joint_actions[self.metaAgentID][i]))
                                obs, reward, done, _ = self.env.step(i-1, Action(joint_actions[self.metaAgentID][i]).value)
                                #print("After agent: ", i-1, " goal: ", self.env.agent_goal[i-1], " state: ", self.env.agent_state[i-1], " action: ", Action(joint_actions[self.metaAgentID][i]), " goal_vector: ", self.goal_vector_calc(i))
                                joint_observations[self.metaAgentID][i] = obs
                                joint_rewards[self.metaAgentID][i]      = reward
                                joint_done[self.metaAgentID][i]         = done
                            if saveGIF and self.agentID == 1:
                                GIF_frames.append(self.env.render(zoom_size=30, agent_id=None))

                        self.synchronize()  # synchronize threads

                        # Get observation,reward, valid actions for each agent 
                        s1           = joint_observations[self.metaAgentID][self.agentID]
                        s1 = s1[1:5]
                        #s1[0] = self.env.obstacle_map #Substitute agent map for obstacle map
                        r            = copy.deepcopy(joint_rewards[self.metaAgentID][self.agentID]) 
                        validActions = self.env.listValidActions(self.agentID-1)

                        self.synchronize() 
                        # Append to Appropriate buffers 
                        if not skipping_state :
                            episode_buffer.append([s, a, joint_rewards[self.metaAgentID][self.agentID] , s1, v[0, 0], train_valid, goal_vector, train_val,train_policy])
                            episode_values.append(v[0, 0])
                        episode_reward += r
                        episode_step_count += 1

                        # Update State
                        s = s1
                        goal_vector = self.goal_vector_calc(self.agentID)

                        # If the episode hasn't ended, but the experience buffer is full, then we
                        # make an update step using that experience rollout.
                        if (len(episode_buffer)>1) and ((len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0) or joint_done[self.metaAgentID][self.agentID] or episode_step_count==max_episode_length):
                            # Since we don't know what the true final return is,
                            # we "bootstrap" from our current value estimation.
                            if len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE:
                                train_buffer = episode_buffer[-EXPERIENCE_BUFFER_SIZE:]
                            else:
                                train_buffer = episode_buffer[:]

                            if joint_done[self.metaAgentID][self.agentID]:
                                s1Value        = 0       # Terminal state
                                episode_buffer = []
                                joint_done[self.metaAgentID][self.agentID] = False
                                targets_done   += 1

                            else:
                                s1Value = self.sess.run(self.local_AC.value,
                                                   feed_dict={self.local_AC.inputs     : np.array([s]),
                                                              self.local_AC.goal_pos   : [goal_vector],
                                                              self.local_AC.state_in[0]: rnn_state[0],
                                                              self.local_AC.state_in[1]: rnn_state[1]})[0, 0]


                            self.loss_metrics, grads = self.calculateGradient(train_buffer, s1Value, episode_count, rnn_state0)

                            self.allGradients.append(grads)

                            rnn_state0 = rnn_state

                        self.synchronize()

                        # finish condition: reach max-len or all agents are done under one-shot mode
                        if episode_step_count >= max_episode_length:
                            break

                        
                    episode_lengths[self.metaAgentID].append(episode_step_count)
                    episode_mean_values[self.metaAgentID].append(np.nanmean(episode_values))
                    episode_invalid_ops[self.metaAgentID].append(episode_inv_count)
                    episode_stop_ops[self.metaAgentID].append(episode_stop_count)
                    swarm_reward[self.metaAgentID] += episode_reward
                    swarm_targets[self.metaAgentID] += targets_done


                    self.synchronize()
                    if self.agentID == 1:
                        episode_rewards[self.metaAgentID].append(swarm_reward[self.metaAgentID])
                        episode_finishes[self.metaAgentID].append(swarm_targets[self.metaAgentID])

                        if saveGIF:
                            #make_gif(np.array(GIF_frames),'{}/episode_{:d}_{:d}_{:.1f}.gif'.format(gifs_path,GIF_episode, episode_step_count,swarm_reward[self.metaAgentID]))
                            make_gif(GIF_frames, '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(gifs_path,GIF_episode, episode_step_count,swarm_reward[self.metaAgentID]))

                    self.synchronize()


                    perf_metrics = np.array([
                        episode_step_count,
                        np.nanmean(episode_values),
                        episode_inv_count,
                        episode_stop_count,
                        episode_reward,
                        targets_done
                    ])

                    assert len(self.allGradients) > 0, 'Empty gradients at end of RL episode?!'
                    return perf_metrics


    
    def synchronize(self):
        # handy thing for keeping track of which to release and acquire
        if not hasattr(self, "lock_bool"):
            self.lock_bool = False
        self.groupLock.release(int(self.lock_bool), self.name)
        self.groupLock.acquire(int(not self.lock_bool), self.name)
        self.lock_bool = not self.lock_bool


    
    def work(self, currEpisode, coord, saver, allVariables):
        '''
        Interacts with the environment. The agent gets either gradients or experience buffer
        '''
        self.currEpisode = currEpisode
        

        if COMPUTE_TYPE == COMPUTE_OPTIONS.multiThreaded:
            self.perf_metrics = self.run_episode_multithreaded(currEpisode, coord)
        else:
            print("not implemented")
            assert(1==0)            


        # gradients are accessed by the runner in self.allGradients
        return



    # Used for imitation learning
    def parse_path(self,episode_count):
        """needed function to take the path generated from M* and create the
        observations and actions for the agent
        path: the exact path ouput by M*, assuming the correct number of agents
        returns: the list of rollouts for the "episode":
                list of length num_agents with each sublist a list of tuples
                (observation[0],observation[1],optimal_action,reward)"""
        
        result           =[[] for i in range(self.num_workers)]
        actions          ={} 
        o                ={}
        train_imitation  ={} 
        targets_done     = 0 
        saveGIF          = False 
        all_obs = []
        all_goal_vectors = []
        goal_vector = {}

        if np.random.rand() < IL_GIF_PROB : 
            saveGIF    =True     
        if saveGIF and OUTPUT_IL_GIFS:
            GIF_frames = [self.env.render(zoom_size=30, agent_id=None)] 

        single_done    = False 
        new_call       = False 
        new_MSTAR_call = False 

        for agentID in range(1, self.num_workers + 1):
            o[agentID] = self.env._observe(agentID-1)
            #print(np.shape(o[agentID][1:5]))
            #o[agentID][0] = self.env.obstacle_map
            o[agentID] = o[agentID][1:5]
            goal_vector[agentID] = self.goal_vector_calc(agentID)
            train_imitation[agentID] = 1 
        step_count = 0
        while step_count <= IL_MAX_EP_LENGTH :
            path = self.expert_until_first_goal()
            if path is None:  # solution not exists
                if step_count !=0 :
                    return result,targets_done
                #print('Failed intially')     
                return None,0 
            none_on_goal = True
            path_step = 1
            while none_on_goal and step_count <= IL_MAX_EP_LENGTH:
                completed_agents =[] 
                start_positions =[] 
                goals =[] 
                
                for i in range(self.num_workers):
                    agent_id = i+1
                    next_pos = path[path_step][i]
                    diff = self.tuple_minus(next_pos, self.env.agent_state[agent_id-1])
                    
                    
                    actions[agent_id] = self.env.dir2action(diff)

                repeat_agents = list(range(self.num_workers))
                #print(path)
                #print(self.env.agent_state)
                #print(actions)
                #print(repeat_agents)
                while repeat_agents:
                    for i in range(self.num_workers) :
                        
                        if(i not in repeat_agents):
                            continue
                        
                        agent_id = i+1
                        
                        pre_state = self.env.agent_state[agent_id-1]
                        obs, r, done, _ = self.env.step(agent_id-1, actions[agent_id].value)
                        if self.env.agent_state[agent_id-1] == pre_state and not actions[agent_id] == Action.NOOP: #this means that the agent couldn't move although it should
                            """
                            print(path)
                            print(actions[agent_id])
                            print(self.env.agent_state[agent_id-1])
                            print(pre_state)
                            print("not done")
                            """
                            continue
                        else:
                            repeat_agents.remove(agent_id-1)
                 
                            
                        #obs[0] = self.env.obstacle_map
                        obs = obs[1:5]
                        goal = self.goal_vector_calc(agent_id)
                        all_obs.append(obs)
                        all_goal_vectors.append(goal)
                        result[i].append([o[agent_id], goal_vector[agent_id], actions[agent_id].value,train_imitation[agent_id]])
                        if done:
                            
                            completed_agents.append(i) 
                            targets_done +=1 
                            single_done = True 
                            if targets_done% MSTAR_CALL_FREQUENCY ==0 :
                                new_MSTAR_call = True 
                            else :     
                                new_call = True 
                #print("there")
                if saveGIF and OUTPUT_IL_GIFS:   
                    GIF_frames.append(self.env.render(zoom_size=30, agent_id=None))     
                if single_done and new_MSTAR_call :
                    path = self.expert_until_first_goal()   
                    if path is None :
                        return result, targets_done
                    path_step = 0 
                elif single_done and new_call : 
                    path = path[path_step:] 
                    path = [ list(state) for state in path ]     
                    for finished_agent in completed_agents :
                        path = merge_plans(path, [None] * len(path), finished_agent) 
                    try :
                        while path[-1]==path[-2] :
                           path = path[:-1]  
                    except :
                        assert(len(path)<=2)        

                    for i in range(1, self.num_workers + 1):
                        start_positions.append(self.env.agent_state[i-1])
                        goals.append(self.env.agent_goal[i])
                    world =self.env.obstacle_map    
                   # print('OLD PATH', path) # print('CURRENT POSITIONS', start_positions) # print('CURRENT GOALS',goals) # print('WORLD',world)
                    try :
                        path = priority_planner(world,tuple(start_positions),tuple(goals),path )
                    except :
                        path = self.expert_until_first_goal()  
                        if path == None :
                                return result,targets_done  
                    path_step = 0
                o = all_obs
                goal_vector = all_goal_vectors
                step_count += 1
                path_step += 1
                new_call = False
                new_MSTAR_call= False  
        if saveGIF and OUTPUT_IL_GIFS:          
            #make_gif(np.array(GIF_frames),'{}/episodeIL_{}.gif'.format(gifs_path,episode_count))                                                                  
            make_gif(GIF_frames, '{}/episodeIL_{}.gif'.format(gifs_path,episode_count))
        return result, targets_done

    
    def shouldRun(self, coord, episode_count=None):
        if TRAINING:
            return not coord.should_stop()


    def expert_until_first_goal(self, inflation=2.0, time_limit=60.0):
    
        world = self.env.obstacle_map
        start_positions = []
        goals = []
        for i in range(1, self.num_workers + 1):
            start_positions.append(self.env.agent_state[i-1])
            goals.append(self.env.agent_goal[i-1])
        mstar_path = None
        start_time = time.time()
       
        try:
            max_time += time_limit

            mstar_path = cpp_mstar.find_path(world, start_positions, goals, inflation, time_limit/5.0)

        except OutOfTimeError:
            # M* timed out
            print("timeout")
            print('World', world)
            print('Start Pos', start_positions)
            print('Goals', goals)
        except NoSolutionError:
            print("nosol????")
            print('World', world)
            print('Start Pos', start_positions)
            print('Goals', goals)

        except:
            c_time = time.time() - start_time
            if c_time > time_limit:
                return mstar_path

            #print("cpp_mstar crash most likely... trying python mstar instead")
            try:
                mstar_path = od_mstar.find_path(world, start_positions, goals,
                                                inflation=inflation, time_limit=time_limit)
            except OutOfTimeError:
                # M* timed out
                print("timeout")
                print('World', world)
                print('Start Pos', start_positions)
                print('Goals', goals)
            except NoSolutionError:
                print("nosol????")
                print('World2', world)
                print('Start Pos2', start_positions)
                print('Goals2', goals)
            except:
                print("Unknown bug?!")

        return mstar_path


    def tuple_minus(self, a, b):
        """ a - b """
        return tuple(map(sub, a, b))
    
    def goal_vector_calc(self, agentID):
        goal_vector = []
        goal_vector.append(self.env.agent_goal[agentID-1][0] - self.env.agent_state[agentID-1][0])
        goal_vector.append(self.env.agent_goal[agentID-1][1] - self.env.agent_state[agentID-1][1])
        goal_vector.append((goal_vector[0] ** 2 + goal_vector[1] ** 2) ** .5)
        
        if goal_vector[2] != 0:
            goal_vector[0] = goal_vector[0] / goal_vector[2]
            goal_vector[1] = goal_vector[1] / goal_vector[2]
        if goal_vector[2] > 60:
            goal_vector[2] = 60
            
        return goal_vector
    
    
def make_gif(frames, fname):
    frames[0].save(fname, save_all=True, append_images=frames[1:], optimize=False, duration=100, loop=0)
