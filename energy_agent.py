import numpy as np
from grid_world import GridworldEnv
import plotting
import itertools
from sarsa import sarsa
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class RBM(object):
    def __init__(self, nagent, nstate, nhid, naction):
        self.env = GridworldEnv()

        self.nstate = nstate
        self.nhid = nhid
        self.naction = naction
        self.nagent = nagent
        scale = 0.01
        #self.state_weights = [scale*np.random.randn(nstate,nhid), scale*np.random.randn(nstate, nhid), scale*np.random.randn(nstate, nhid)]
        #self.state_agent = [np.zeros((nstate, 1)), np.zeros((nstate, 1)), np.zeros((nstate, 1))]

        self.state_weights = [scale*np.random.randn(nstate,nhid)]
        self.state_agent = [np.zeros((nstate, 1))]

        self.action_weights  = [scale * np.random.randn(naction, nhid)]
        self.biases_states = [np.zeros((nstate, 1)), np.zeros((nstate, 1)), np.zeros((nstate, 1))]

        self.biases_action = [np.zeros((naction, 1))]
        self.biases_hidden = np.zeros((nhid, 1))


        self.state_agent[0][5] = 1
        #self.state_agent[1][0] = 1
        #self.state_agent[2][15] = 1
        self.old_state = 5

        self.action_agent = [np.zeros((naction, 1))]
        self.action_agent[0][0] = 1

        self.action_performed = 0
        self.learning_rate = 0.01
        self.hid_state = np.zeros((nhid, 1))

        self.free_energy = 0
        self.temperature = 1
        self.reward_ = 0
        self.discount_factor = 0.1

    def reset(self):
        new_s = np.random.randint(1,15)
        self.learning_rate = 0.01
        self.state_agent[0][new_s] = 1
        #self.state_agent[1][0] = 1
        #self.state_agent[2][15] = 1
        self.old_state = new_s
        self.action_agent[0][0] = 1
        self.action_performed = 0
        self.free_energy = 0
        self.temperature = 1
        self.reward_ = 0
        self.discount_factor = 0.1


    def get_energy_state(self, hid_K, agent_M, state_I):
        '''
        We have 3 agents!
        self.state_weights = [np.empty((nstate, nhid)), np.empty((nstate, nhid)), np.empty((nstate, nhid))]
        '''
        term_a = self.state_weights[agent_M][state_I][hid_K]*self.state_agent[agent_M][state_I] * self.hid_state[hid_K]
        term_b = self.biases_states[agent_M][state_I] * self.state_agent[agent_M][state_I]

        return term_a + term_b

    def get_energy_action(self, hid_K, agent_M, action_I):
        '''
        self.action_weights  = [np.empty((naction, nhid))]
        self.action_agent = [np.zeros((naction, 1))]
        '''
        term_a = self.action_weights[agent_M][action_I][hid_K]* self.action_agent[agent_M][action_I] * self.hid_state[hid_K]
        term_b = self.biases_action[agent_M][action_I] * self.action_agent[agent_M][action_I]
        return term_a + term_b

    def get_second_term(self):
        tot_ = 0
        for i in range(self.nhid):
           tot_ += self.hid_state[i] * self.biases_hidden[i]
        return tot_

    def get_third_term(self):
        tot_ = 0
        for i in range(self.nhid):
            tot_ += self.hid_state[i] * np.log(self.hid_state[i]) + (1 - self.hid_state[i]) *np.log(1 - self.hid_state[i])
        return tot_

    def get_free_energy(self):
        first_term = 0
        second_term = 0
        third_term = 0
        for i in range(self.nhid):
            for j in range(self.nagent):
                for k in range(self.nstate):
                    first_term += self.get_energy_state(i, j, k)

        for i in range(self.nhid):
            for j in range(1):
                for k in range(self.naction):
                    first_term += self.get_energy_action(i, j, k)

        second_term += self.get_second_term()
        third_term += self.get_third_term()
        self.free_energy = first_term + second_term + third_term
        return  self.free_energy

    def _update_action_weights(self, diff):
        '''
        self.action_weights  = [np.empty((naction, nhid))]
        self.action_agent = [np.zeros((naction, 1))]
        '''
        for i in range(self.nhid):
            for j in range(1):
                for k in range(self.naction):
                    self.action_weights[j][k][i] -= self.learning_rate * diff * self.hid_state[i] * self.action_agent[j][k]


    def _update_state_weights(self, diff):
        '''
        self.state_weights = [np.empty((nstate, nhid)), np.empty((nstate, nhid)), np.empty((nstate, nhid))]
        self.state_agent = [np.zeros((nstate, 1)), np.zeros((nstate, 1)), np.zeros((nstate, 1))]
        '''
        for i in range(self.nhid):
            for j in range(self.nagent):
                for k in range(self.nstate):
                    self.state_weights[j][k][i] -= self.learning_rate * diff* self.hid_state[i] * self.state_agent[j][k]


    def _update_hiddens(self):
        for i in range(self.nhid):
            tot_ = 0
            for j in range(self.nagent):
                for k in range(self.nstate):
                    tot_ += self.state_weights[j][k][i]*self.state_agent[j][k]

            for j in range(1):
                for k in range(self.naction):
                    tot_ += self.action_weights[j][k][i]* self.action_agent[j][k]

            tot_ += self.biases_hidden[i]
            self.hid_state[i] = 1 * 1.0/ (1 + np.exp(-1 * tot_))


    def _update_states(self):
        old_state = self.old_state
        action_performed = self.action_performed
        new_state = self.env.P[old_state][action_performed][0][1]
        reward_ = self.env.P[old_state][action_performed][0][2]
        self.state_agent[0][new_state] = 1
        self.old_state = new_state
        self.reward_ = reward_
        return reward_

    def update_action(self, update):
        self.norm_constant = []
        self.free_energy_ = []
        for i in range(self.naction):
            self.action_agent[0] = np.zeros((self.naction, 1))
            self.action_agent[0][i] = 1
            free_energy = self.get_free_energy()
            self.free_energy_.append(free_energy)
            self.norm_constant.append(np.exp(free_energy/self.temperature))
        output = softmax(self.norm_constant)
        action_number = np.argmax(output)
        next_step_free_energy = np.max(self.free_energy_)
        if update:
            #print 'Performing action,', action_number
            self.action_agent[0][action_number] = 1
            self.action_performed = action_number

        return next_step_free_energy

    def gibbs_sampling(self, num_episodes):
        stats = plotting.EpisodeStats(
                 episode_lengths=np.zeros(num_episodes),
                 episode_rewards=np.zeros(num_episodes))
        for i_episode in range(num_episodes):
            self.temperature = 1
            self.reset()
            #self.learning_rate = self.learning_rate * 0.9
            self.temperature =  self.temperature - 0.9/50
            for i in itertools.count():
                self._update_hiddens()
                self.update_action(update=True)
                free_energy_1 = self.get_free_energy()
                reward_ = self._update_states()
                #print 'New State', self.old_state
                stats.episode_rewards[i_episode] += reward_
                stats.episode_lengths[i_episode] = i

                if self.old_state ==15 or self.old_state ==0:
                    print 'Ith episode, episode len', i_episode, i
                    break
                free_energy_2 = self.update_action(update=True)
                diff = reward_ + self.discount_factor * free_energy_2 - free_energy_1
                self._update_action_weights(diff)
                self._update_state_weights(diff)
        return stats


'''
rbm = RBM(nagent=3, nstate=16, nhid=20, naction=4)
stats1 = rbm.gibbs_sampling(100)

rbm = RBM(nagent=3, nstate=16, nhid=50, naction=4)
stats2 = rbm.gibbs_sampling(100)
'''
rbm = RBM(nagent=1, nstate=16, nhid=100, naction=4)
stats3 = rbm.gibbs_sampling(100)

Q, stats4 = sarsa(rbm.env, 100)
plotting.plot_episode_stats(stats3, stats4)
import ipdb
ipdb.set_trace()
