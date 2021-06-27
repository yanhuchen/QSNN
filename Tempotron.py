import numpy as np


class Tempotron:
    """
    A class representing a tempotron, as described in
    Gutig & Sompolinsky (2006).
    The (subthreshold) membrane voltage of the tempotron
    is a weighted sum from all incoming spikes and the
    resting potential of the neuron. The contribution of
    each spike decays exponentiall with time, the speed of
    this decay is determined by two parameters tau and tau_s,
    denoting the decay time constants of membrane integration
    and synaptic currents, respectively.
    """
    def __init__(self, V_rest, tau, tau_s, train_num, test_num, steps, synaptic_efficacies,str_len):
        # tempotron所需要使用到的一些超参数
        self.V_rest = V_rest #静息电位
        self.tau = float(tau)
        self.tau_s = float(tau_s)
        self.log_tts = np.log(self.tau/self.tau_s)
        self.threshold = 1 * self.tau * self.tau_s / (self.tau-self.tau_s)#阈值
        #self.threshold = 10
        self.efficacies = synaptic_efficacies #突触的权重
        self.train_num = train_num #训练集数据样本个数
        self.test_num = test_num #测试集数据样本个数
        self.steps = steps#迭代次数
        self.str_len = str_len
        
        #我们希望保留的计算值
        #每一组脉冲到达最大电位的时间 tmax
        self.all_tmax = np.zeros(self.train_num)
        #每一组脉冲到达的最大电位 vmax
        self.all_vmax = np.zeros(self.train_num)
        
        #测试集
        self.test_all_tmax = np.zeros(self.test_num)
        self.test_all_vmax = np.zeros(self.test_num)
        self.pre_train_label = np.zeros(self.train_num) 
        self.pre_test_label = np.zeros(self.test_num) 
        
        #所有迭代次数的准确率
        self.percentage = np.zeros(steps)
        self.test_percentage = np.zeros(steps)
        
        # compute normalisation factor V_0
        self.V_norm = self.compute_norm_factor(tau, tau_s)
    
        
    def compute_norm_factor(self, tau, tau_s):
        """
        Compute and return the normalisation factor:
        V_0 = (tau * tau_s * log(tau/tau_s)) / (tau - tau_s)
        That normalises the function:
                
        K(t-t_i) = V_0 (exp(-(t-t_i)/tau) - exp(-(t-t_i)/tau_s)
        Such that it amplitude is 1 and the unitary PSP
        amplitudes are given by the synaptic efficacies.
        """
        tmax = (tau * tau_s * np.log(tau/tau_s)) / (tau - tau_s)
        v_max = self.K(1, tmax, 0)
        V_0 = 1/v_max
        return V_0
        
    def K(self, V_0, t, t_i):
        """
        Compute the function
        K(t-t_i) = V_0 (exp(-(t-t_i)/tau) - exp(-(t-t_i)/tau_s)
        """
        if t < t_i:
            value = 0
        else:
            value = V_0 * (np.exp(-(t-t_i)/self.tau) - np.exp(-(t-t_i)/self.tau_s))
        return value
    
    def train(self, io_pairs, learning_rate, step):
        """
        Train the tempotron on the given input-output pairs,
        applying gradient decscend to adapt the weights.
        :param steps: the maximum number of training steps
        :param io_pairs: a list with tuples of spike times and the
                         desired response on them
        :param learning_rate: the learning rate of the gradient descend
        """
        # Run until maximum number of steps is reached or
        # no weight updates occur anymore

        self.correct = 0
        j=0
        for spike_times, target in io_pairs:
            self.all_tmax[j], self.all_vmax[j] = self.adapt_weights(spike_times, target, learning_rate, j)
            j=j+1
        #计算准确率
        self.percentage[step] = self.correct/self.train_num
        
        return self.percentage[step]
            
    def adapt_weights(self, spike_times, target, learning_rate, j):
        """
        Modify the synaptic efficacies such that the learns
        to classify the input pattern correctly.
        Whenever an error occurs, the following update is
        computed:
        dw = lambda sum_{ti} K(t_max, ti)
        The synaptic efficacies are increased by this weight
        if the tempotron did erroneously not elecit an output
        spike, and decreased if it erroneously did.
        :param spike_times: an array with lists of spike times
                            for every afferent
        :param output_spike: the classification of the input pattern
        :type output_spike: Boolean
        """

        # compute tmax
        tmax = self.compute_tmax(spike_times)
        vmax = self.compute_membrane_potential(tmax, spike_times)#计算膜电位

        # print "vmax = ", vmax
        # print "target = ", target

        # if target output is correct, don't adapt weights
        if (vmax >= self.threshold) == target:
            # print "no weight update necessary"
            self.correct = self.correct + 1
            self.pre_train_label[j] = target
            
        else:
            # compute weight updates
            dw = self.dw(learning_rate, tmax, spike_times)
            # print "update =", dw
    
            if target is True:
                self.efficacies += dw
                self.pre_train_label[j] = False
            else:
                self.efficacies -= dw
                self.pre_train_label[j] = True
            
        return tmax, vmax
            
    def compute_tmax(self, spike_times):
        """
        Compute the maximum mebrane potential of the tempotron as
        a result of the input spikes.
        The maxima of the function can be computed analytically, but as
        there are as many maxima and minima as their are number of spikes,
        we still need to sort through them to find the highest one.
        The maxima are given by:
        t = (log(tau/tau_s) + log(sum w_n exp(t_n/tau_s)) - log(sum w_n exp(t_n/tau)))*tau_s*tau/ (tau-tau_s)
        for n = 1, 2, ..., len(spike_times)
        The time at which the membrane potential is maximal is given by
        Check if the input spikes result produce the desired
        output. Return tmax. (maybe I should return something else)
        计算V(t)在什么时刻到达最大值，对V(t)求导，该函数是计算dV/dt=0时的结果
        tmax == (log(tau/tau_s) + log(sum w_n exp(t_n/tau_s)) - log(sum w_n exp(t_n/tau)))*tau_s*tau/ (tau-tau_s)
        但是由于公式中包含指数关系，在计算过程中极有可能超出数据类型所能容纳的极限，
        这里可以把它们先放在指数上计算，然后再最后统一给出
        """

        # sort spikes in chronological order
        spikes_chron = [(time, synapse) for synapse in range(len(spike_times)) for time in spike_times[synapse]]
        spikes_chron.sort()

        # Make a list of spike times and their corresponding weights
        spikes = [(s[0], self.efficacies[s[1]]) for s in spikes_chron]
        times = np.array([spike[0] for spike in spikes]) /2**self.str_len #将毫秒变成秒
        weights = np.array([spike[1] for spike in spikes])
        weights = np.exp(weights)
        
        #和突触等长的sum_tau和sum_tau_s
        sum_tau = (weights*np.exp(times/self.tau)).cumsum()
        sum_tau_s = (weights*np.exp(times/self.tau_s)).cumsum()

        # when an inhibitive spike is generated when the membrane potential
        # is still growing, the derivative does not exist in the maximum
        # In such cases, thus when sum_tau/sum_tau_s is negative,
        # manually set tmax to the spike time of the second spike
        div = sum_tau_s/sum_tau
        boundary_cases = div < 0
        div[boundary_cases] = 10

        #tmax_list = self.tau*self.tau_s*(self.log_tts + np.log(div))/(self.tau - self.tau_s)
        tmax_list = self.tau*self.tau_s*(self.log_tts + np.log(sum_tau_s) - np.log(sum_tau))/(self.tau - self.tau_s)
        tmax_list[boundary_cases] = times[boundary_cases]

        vmax_list = np.array([self.compute_membrane_potential(t, spike_times) for t in tmax_list])

        tmax = tmax_list[vmax_list.argmax()]
        
        return tmax
    
    def compute_membrane_potential(self, t, spike_times):
        """
        Compute the membrane potential of the neuron given
        by the function:
        V(t) = sum_i w_i sum_{t_i} K(t-t_i) + V_rest
        Where w_i denote the synaptic efficacies and t_i denote
        ith afferent.
        
        :param spike_times: an array with at position i the spike times of
                            the ith afferent
        :type spike_times: numpy.ndarray
        """
        # create an array with the contributions of the
        # spikes for each synaps
        spike_contribs = self.compute_spike_contributions(t, spike_times)

        # multiply with the synaptic efficacies
        total_incoming = spike_contribs * self.efficacies

        # add sum and add V_rest to get membrane potential
        V = total_incoming.sum() + self.V_rest
        
        return V
    
    
    def dw(self, learning_rate, tmax, spike_times):
        """
        Compute the update for synaptic efficacies wi,
        according to the following learning rule
        (implementing gradient descend dynamics):
        dwi = lambda sum_{ti} K(t_max, ti)
        where lambda is the learning rate and t_max denotes
        the time at which the postsynaptic potential V(t)
        reached its maximal value.
        """
        # compute the contributions of the individual spikes at
        # time tmax
        spike_contribs = self.compute_spike_contributions(tmax, spike_times)

        # multiply with learning rate to get updates
        update = learning_rate * spike_contribs

        return update
    
    
    def compute_spike_contributions(self, t, spike_times):
        """
        Compute the decayed contribution of the incoming spikes.
        """
        # nr of synapses
        N_synapse = len(spike_times)
        # loop over spike times to compute the contributions
        # of individual spikes
        spike_contribs = np.zeros(N_synapse)
        for neuron_pos in range(N_synapse):
            for spike_time in spike_times[neuron_pos]:
                # print self.K(self.V_rest, t, spike_time)
                spike_contribs[neuron_pos] += self.K(self.V_norm, t, spike_time)
        return spike_contribs
    
    
    def test(self, io_pairs, step):
        test_correct = 0
        j = 0
        for spike_times, target in io_pairs:
             tmax = self.compute_tmax(spike_times)
             vmax = self.compute_membrane_potential(tmax, spike_times)#计算膜电位
             self.test_all_tmax[j] = tmax
             self.test_all_vmax[j] = vmax
             
             #判断测试集的分类准确率
             if (vmax >= self.threshold) == target:
                # print "no weight update necessary"
                test_correct = test_correct + 1
                self.pre_test_label[j] = target
             else:
                self.pre_test_label[j] = not target
             j = j+1
        self.test_percentage[step] = test_correct / len(io_pairs)
        
        return test_correct / len(io_pairs)
        