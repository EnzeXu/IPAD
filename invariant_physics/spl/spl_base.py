import os
import ast
import pickle
from collections import defaultdict
from functools import partial

import numpy as np

from .production_rule_utils import get_nonterminal_rules, preprocess_exp, simplify_eqs, to_prod
from ..dataset import TermTrace
# from ..spl import purify_strategy
from ..dataset import extract, evaluate_expression

def save_reward_history(i_episode, eqs, reward, path):
    with open(path, "a") as f:
        f.write(f"{i_episode},{','.join(eqs)},{reward}\n")


def len_tree_size(string: str, strategy="default"):
    assert strategy in ["default", "shorten"]
    if strategy == "default":
        return 1
    target_list = ["A", "B"]
    assert "->" in string
    target = string.split("->")[-1]
    # print([target.count(ch) for ch in target_list])
    count_sum = sum([target.count(ch) for ch in target_list])
    if count_sum >= 2 or ("B" in target and "C" in target):
        return 1
    else:
        return 0


class SplBase:
    
    def __init__(self, data_sample, base_grammars, aug_grammars, nt_nodes, max_len, max_module, aug_grammars_allowed,
                 func_score, eta, exploration_rate=1/np.sqrt(2), max_added_grammar_count=2, force=True,
                 added_basic_grammars=[], forced_nodes=[], data_t_series_list=None, i_transplant=1, i_test=1, output_dir='.', task="", num_env=None, term_trace_path=None,
                 variable_list=None, full_data=None, timestring=None, tree_size_strategy=None):
        self.data_sample = data_sample
        self.base_grammars = base_grammars
        self.grammars = base_grammars + [x for x in aug_grammars if x not in base_grammars] 
        self.nt_nodes = nt_nodes
        self.max_len = max_len
        self.max_module = max_module
        self.max_aug = aug_grammars_allowed
        self.good_modules = []
        self.score = func_score
        self.exploration_rate = exploration_rate
        self.UCBs = defaultdict(partial(np.zeros, len(self.grammars)))
        self.QN = defaultdict(partial(np.zeros, 2))
        self.scale = 0
        self.eta = eta
        self.forced_nodes = forced_nodes
        self.added_basic_grammars = added_basic_grammars
        self.max_added_grammar_count = max_added_grammar_count
        self.force = force
        self.grammars = self.grammars + [x for x in self.added_basic_grammars if x not in self.grammars]
        self.data_t_series_list = data_t_series_list

        self.current_episode = 0
        self.i_transplant = i_transplant
        self.i_test = i_test
        self.output_dir = output_dir
        self.task = task
        # search history
        self.states = []
        self.reward_his = []
        self.best_solution = ('nothing', 0)        

        self.num_env = num_env
        self.term_trace_path = term_trace_path
        self.variable_list = variable_list
        self.full_data = full_data

        self.timestring = timestring
        self.tree_size_strategy = tree_size_strategy

        

    def valid_prods(self, Node):
        """
        Get index of all possible production rules starting with a given node
        """
        return [self.grammars.index(x) for x in self.grammars if x.startswith(Node)] 
    
    
    def tree_to_eq(self, prods):
        """
        Convert a parse tree to equation form
        """
        # print(f"Tree: prods =", prods)
        seq = ['f']
        for prod in prods:
            if str(prod[0]) == 'Nothing':
                break
            for ix, s in enumerate(seq):
                if s == prod[0]:
                    seq = seq[:ix] + list(prod[3:]) + seq[ix+1:]
                    break
        try:
            return ''.join(seq)
        except:
            return ''


    def state_to_seq(self, state):
        """
        Convert the state to sequence of index
        """
        aug_grammars = ['f->A'] + self.grammars
        seq = np.zeros(self.max_len)
        prods = state.split(',')
        for i, prod in enumerate(prods):
            seq[i] = aug_grammars.index(prod)
        return seq
    
    
    def state_to_onehot(self, state):
        """
        Convert the state to one hot matrix 
        """
        aug_grammars = ['f->A'] + self.grammars
        state_oh = np.zeros([self.max_len, len(aug_grammars)])
        prods = state.split(',')
        for i in range(len(prods)):
            state_oh[i, aug_grammars.index(prods[i])] = 1
            
        return state_oh

        
    def get_ntn(self, prod, prod_idx):
        """
        Get all the non-terminal nodes from right-hand side of a production rule grammar
        """
        if prod_idx >= len(self.base_grammars): return []
        else: return [i for i in prod[3:] if i in self.nt_nodes]


    def get_unvisited(self, state, node):
        """
        Get index of all unvisited child
        """
        valid_action = self.valid_prods(node)
        return [a for a in valid_action if self.QN[state + ',' + self.grammars[a]][1] == 0]
    
            
    def print_solution(self, solu, i_episode):
        # print('Episode', i_episode, solu)
        pass
    
    def step(self, state, action_idx, ntn):
        """
        state: all production rules
        action_idx: index of grammar starts from the current Non-terminal Node
        tree: the current tree
        ntn: all remaining non-terminal nodes
        
        
        This defines one step of Parse Tree traversal
        return tree (next state), remaining non-terminal nodes, reward, and if it is done
        """
        action = self.grammars[action_idx]
        state = state + ',' + action
        # print(f"[debug] state = {state}")
        ntn = self.get_ntn(action, action_idx) + ntn[1:]
        
        if not ntn:
            tree_size = sum([len_tree_size(state_string, self.tree_size_strategy) for state_string in state.split(',')])
            # len(state.split(',')),
            reward, eqs = self.score(self.tree_to_eq(state.split(',')), tree_size,
                                    self.data_sample, eta=self.eta, data_t_series_list=self.data_t_series_list, variable_list=self.variable_list)
            return state, ntn, reward, True, eqs
        else:
            return state, ntn, 0, False, None
    

    def rollout(self, num_play, state_initial, ntn_initial):
        """
        Perform a n-play rollout simulation, get the maximum reward
        """
        best_eq = ''
        best_r = 0
        for n in range(num_play):
            done = False
            state = state_initial
            ntn = ntn_initial

            while not done:
                valid_index = self.valid_prods(ntn[0])
                action = np.random.choice(valid_index)
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn)
                state = next_state
                ntn = ntn_next
    
                if state.count(',') >= self.max_len:
                    break
            
            if done:
                if reward > best_r:
                    self.update_modules(next_state, reward, eq)
                    best_eq = eq
                    best_r = reward
                    
        return best_r, best_eq
    

    def update_ucb_mcts(self, state, action):
        """
        Get the ucb score for a given child of current node
        """
        next_state = state + ',' + action
        Q_child = self.QN[next_state][0]
        N_parent = self.QN[state][1]
        N_child = self.QN[next_state][1]
        return Q_child / N_child + self.exploration_rate*np.sqrt(np.log(N_parent) / N_child)
        # return Q_child / N_child + self.exploration_rate*np.sqrt(np.sqrt(N_parent) / N_child)


    def update_QN_scale(self, new_scale):
        """
        Update the Q values self.scaled by the new best reward.
        """

        if self.scale != 0:
            for s in self.QN:
                self.QN[s][0] *= (self.scale / new_scale)

        self.scale = new_scale
        
        
    def backpropogate(self, state, action_index, reward):
        """
        Update the Q, N and ucb for all corresponding decedent after a complete rollout
        """

        action = self.grammars[action_index]
        if self.scale != 0 : self.QN[state + ',' + action][0] += reward / self.scale
        else: self.QN[state + ',' + action][0] += 0
        self.QN[state + ',' + action][1] += 1
        
        while state:
            if self.scale != 0 : self.QN[state][0] += reward / self.scale
            else: self.QN[state][0] += 0
            self.QN[state][1] += 1
            if len(self.UCBs[state]) < len(self.grammars):
                self.UCBs[state] = np.concatenate([self.UCBs[state], np.zeros(len(self.grammars) - len(self.UCBs[state]))])
            self.UCBs[state][self.grammars.index(action)] = self.update_ucb_mcts(state, action)
            if ',' in state: state, action = state.rsplit(',', 1)
            else: state = ''
    
    
    def policy1(self, state, node):
        nA = len(self.grammars)
        valid_action = self.valid_prods(node)

        # collect ucb scores for all valid actions
        policy_valid = []

        sum_ucb = sum(self.UCBs[state][valid_action])

        for a in valid_action:
            policy_mcts = self.UCBs[state][a] / sum_ucb
            policy_valid.append(policy_mcts)

        # if all ucb scores identical, return uniform policy
        if len(set(policy_valid)) == 1:
            A = np.zeros(nA)
            A[valid_action] = float(1 / len(valid_action))
            return A

        # return action with largest ucb score
        A = np.zeros(nA, dtype=float)  
        best_action = valid_action[np.argmax(policy_valid)]
        A[best_action] += 0.8
        A[valid_action] += float(0.2 / len(valid_action))
        return A
    
    def policy2(self, UC):
        nA = len(self.grammars)
        if len(UC) != len(set(UC)):
            # print(UC)
            # print(self.grammars)
            pass
        A = np.zeros(nA, dtype=float)  
        A[UC] += float(1 / len(UC))
        return A
    
    def convert_eq_to_tree_forced_node(self, eqs):
        added_basic_grammars = []
        reinsert_node = None
        try:
            for eq in simplify_eqs(eqs):
                exp = ast.parse(eq, "", "eval")
                exp = preprocess_exp(exp)
                nt_rules = get_nonterminal_rules(self.base_grammars)
                t_rules = [r for r in (self.base_grammars + self.added_basic_grammars) if r not in nt_rules]
                prods = to_prod(exp, 'A', t_rules, nt_rules)
                if reinsert_node == None or len(reinsert_node) > len(prods):
                    # find smallest set of production rules for new grammar
                    added_grammar_count = len(added_basic_grammars)
                    for prod in prods:
                        if "-&>" in prods:
                            added_grammar_count += 1
                    if added_grammar_count > self.max_added_grammar_count:
                        continue
                    else:
                        reinsert_node = prods


            added_basic_grammars = [p.replace("-&>", "->") for p in reinsert_node if "-&>" in p]
            added_basic_grammars = list(set(added_basic_grammars))
            reinsert_node = [p.replace("-&>", "->") for p in reinsert_node]
        except Exception as e:
            print(str(e))

        if reinsert_node != None:
            # print("Adding basic grammars:", added_basic_grammars)
            # print("Reinsert node forced:", reinsert_node)
            self.grammars += added_basic_grammars
            self.added_basic_grammars += added_basic_grammars
            self.forced_nodes.append(reinsert_node)

    def update_modules(self, state, reward, eq):
        """
        If we pass by a concise solution with high score, we store it as an 
        single action for future use. 
        """
        module = state[5:]
        if state.count(',') <= self.max_module:
            if not self.good_modules:
                self.good_modules = [(module, reward, eq)]
            elif eq not in [x[2] for x in self.good_modules]:
                if len(self.good_modules) < self.max_aug:
                    self.good_modules = sorted(self.good_modules + [(module, reward, eq)], key = lambda x: x[1])
                else:
                    if reward > self.good_modules[0][1]:
                        self.good_modules = sorted(self.good_modules[1:] + [(module, reward, eq)], key = lambda x: x[1])



    def run(self, num_episodes, num_play=50, print_flag=False, print_freq=50):
        """
        Monte Carlo Tree Search algorithm
        """
        
        nA = len(self.grammars)
        
        # The policy we're following: 

        # policy1 for fully expanded node and policy2 for not fully expanded node

    
        reward_his = []
        best_solution = ('nothing', 0)

        tt = TermTrace(self.num_env)

        for i_episode in range(self.current_episode+1, num_episodes+1):
            if (i_episode) % print_freq == 0 and print_flag:
                # print("\rEpisode {}/{}, current best reward {}\nCurrent grammars:{}.".format(i_episode, num_episodes, best_solution[1], self.grammars), end="")
                # sys.stdout.flush()
                pass
            

            state = 'f->A'
            ntn = ['A']
            UC = self.get_unvisited(state, ntn[0])

            ##### check scenario: if parent node fully expanded or not ####
            
            # scenario 3: if there are forced-to-insert nodes
            if self.forced_nodes:
                for i, grammar in enumerate(self.forced_nodes[-1]):
                    if grammar not in self.grammars:
                        continue
                    action = self.grammars.index(grammar)
                    next_state, ntn_next, reward, done, eqs = self.step(state, action, ntn)


                    if state not in self.states:
                        self.states.append(state)

                    if not done:
                        assert i < len(self.forced_nodes[-1]) - 1, f"Should have reached terminal node for: {self.forced_nodes[-1]}, got {next_state}"
                        state = next_state
                        ntn = ntn_next
                        UC = self.get_unvisited(state, ntn[0])

                        if state.count(',') >= self.max_len:
                            UC = []
                            self.backpropogate(state, action, 0)
                            self.reward_his.append(self.best_solution[1])
                            # print("1 self.best_solution:", self.best_solution[0])
                            if len(self.best_solution) > 0 and len(self.best_solution[0]) == self.num_env:
                                tt.add_iteration_result(self.best_solution[0])
                                save_reward_history(i_episode, self.best_solution[0], self.best_solution[1],
                                                    f"outputs_reward_his/{self.timestring}.csv")
                            break
                    else:
                        UC = []
                        if reward > self.best_solution[1]:
                            self.update_modules(next_state, reward, eqs)
                            self.update_QN_scale(reward)
                            self.best_solution = (eqs, reward)
                            # print(f"new best solution: {self.best_solution}")

                        self.backpropogate(state, action, reward)
                        self.reward_his.append(self.best_solution[1])
                        # print("2 self.best_solution:", self.best_solution[0])
                        if len(self.best_solution) > 0 and len(self.best_solution[0]) == self.num_env:
                            tt.add_iteration_result(self.best_solution[0])
                            save_reward_history(i_episode, self.best_solution[0], self.best_solution[1],
                                                f"outputs_reward_his/{self.timestring}.csv")
                        self.forced_nodes.pop()
                        break

            # scenario 1: if current parent node fully expanded, follow policy1
            nA = len(self.grammars)
            while not UC:
                action = np.random.choice(np.arange(nA), p=self.policy1(state, ntn[0]))
                next_state, ntn_next, reward, done, eqs = self.step(state, action, ntn)
                if state not in self.states:
                    self.states.append(state)

                if not done:
                    state = next_state
                    ntn = ntn_next
                    UC = self.get_unvisited(state, ntn[0])

                    if state.count(',') >= self.max_len:
                        UC = []
                        self.backpropogate(state, action, 0)
                        self.reward_his.append(self.best_solution[1])
                        # print("3 self.best_solution:", self.best_solution[0])
                        if len(self.best_solution) > 0 and len(self.best_solution[0]) == self.num_env:
                            tt.add_iteration_result(self.best_solution[0])
                            save_reward_history(i_episode, self.best_solution[0], self.best_solution[1],
                                                f"outputs_reward_his/{self.timestring}.csv")
                        break
                else:
                    UC = []
                    if reward > self.best_solution[1]:
                        self.update_modules(next_state, reward, eqs)
                        self.update_QN_scale(reward)
                        self.best_solution = (eqs, reward)
                        # save_reward_history(eqs, reward, f"outputs_reward_his/{self.timestring}.csv")
                        if self.force:
                            self.convert_eq_to_tree_forced_node(eqs)

                    self.backpropogate(state, action, reward)
                    # When a node is terminal, the reward is deterministic
                    # So set UCB to 0 to avoid choosing this action again
                    # To avoid division by 0 when all leaves are terminal, we use 1e-6
                    self.UCBs[state][action] = 1e-6 
                    self.reward_his.append(self.best_solution[1])
                    # print("4 self.best_solution:", self.best_solution)
                    if len(self.best_solution) > 0 and len(self.best_solution[0]) == self.num_env:
                        tt.add_iteration_result(self.best_solution[0])
                        save_reward_history(i_episode, self.best_solution[0], self.best_solution[1], f"outputs_reward_his/{self.timestring}.csv")
                    break
                    
            # scenario 2: if current parent node not fully expanded, follow policy2
            if UC:
                action = np.random.choice(np.arange(nA), p=self.policy2(UC))
                next_state, ntn_next, reward, done, eqs = self.step(state, action, ntn)   
                if not done:
                    reward, eqs = self.rollout(num_play, next_state, ntn_next)
                    if state not in self.states:
                        self.states.append(state)

                if reward > self.best_solution[1]:
                    self.update_QN_scale(reward)
                    self.best_solution = (eqs, reward)
                    if self.force:
                        self.convert_eq_to_tree_forced_node(eqs)

                self.backpropogate(state, action, reward)

                self.reward_his.append(self.best_solution[1])
                # print("5 self.best_solution:", self.best_solution[0])
                if len(self.best_solution) > 0 and len(self.best_solution[0]) == self.num_env:
                    tt.add_iteration_result(self.best_solution[0])
                    save_reward_history(i_episode, self.best_solution[0], self.best_solution[1],
                                        f"outputs_reward_his/{self.timestring}.csv")

            if (i_episode) % print_freq == 0:
                # Print & Save object
                if print_flag:
                    print("\rEpisode {}/{}, current best reward {}, Current grammars(n):{}.".format(i_episode, num_episodes, self.best_solution[1], self.grammars))
                    # sys.stdout.flush()
                    pass
                self.current_episode = i_episode
                # Skip .pkl save
#                 num_env = len(self.data_sample)
#                 save_path = os.path.join(self.output_dir, self.task, f"splbase_{num_env}_{self.eta}_{self.i_transplant}_{self.i_test}_min.pkl")
# #                 print(save_path, self.output_dir, self.task)
#                 if not os.path.exists(os.path.dirname(save_path)):
#                     os.makedirs(os.path.dirname(save_path))
#                 with open(save_path, "wb") as f:
#                     pickle.dump(self, f)
#                 pass
        
        tt.draw_term_trace(self.term_trace_path)
        return self.reward_his, self.best_solution, self.good_modules



# def purify_strategy(eq, data, variable_list, threshold=0.03):
#     # data is in shape (N, m). Here m is the dimension of the ODE system
#     # print(f"input: {eq}")
#     full_terms, terms, _ = extract(eq)
#     # print(full_terms)
#     # print(terms)
#     n = data.shape[0]
#     abs_value_array = np.zeros([n, len(full_terms)])
#     abs_ratio_array = np.zeros([n, len(full_terms)])
#     for i in range(n):
#         for j, one_full_term in enumerate(full_terms):
#             abs_value_array[i][j] = np.abs(evaluate_expression(one_full_term, variable_list, data[i]))
#         for j in range(len(full_terms)):
#             abs_ratio_array[i][j] = abs_value_array[i][j] / np.sum(abs_value_array[i])
#     avg_ratio = np.average(abs_ratio_array, axis=0)
#     purified_full_terms = [full_terms[i] for i in range(len(full_terms)) if avg_ratio[i] >= threshold]
#     purified_eq = sp.sympify(sp.Add(*purified_full_terms))
#     # print(avg_ratio)
#     # print(f"output_v20240301: {purified_eq}")
#     return purified_eq
# >>>>>>> main
