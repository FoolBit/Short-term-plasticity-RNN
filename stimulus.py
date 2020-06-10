import numpy as np
import matplotlib.pyplot as plt
from parameters import *


class Stimulus:

    def __init__(self):

        # generate tuning functions
        self.motion_tuning, self.fix_tuning, self.rule_tuning = self.create_tuning_functions()


    def generate_trial(self, test_mode = False, set_rule = None):

        if par['trial_type'] == 'WM':
            trial_info = self.generate_WM_trial(test_mode)

        # input activity needs to be non-negative
        trial_info['neural_input'] = np.maximum(0., trial_info['neural_input'])

        return trial_info

    def generate_WM_trial(self, test_mode):

        """
        Generate a trial based on "Reactivation of latent working memories with transcranial magnetic stimulation"

        Trial outline
        1. Dead period
        2. Fixation
        3. Two sample stimuli presented
        4. Delay (cue in middle, and possibly probe later)
        5. Test stimulus (to cued modality, match or non-match)
        6. Delay (cue in middle, and possibly probe later)
        7. Test stimulus

        INPUTS:
        1. sample_time (duration of sample stimlulus)
        2. test_time
        3. delay_time
        4. cue_time (duration of rule cue, always presented halfway during delay)
        5. probe_time (usually set to one time step, always presented 3/4 through delay

        总体流程：
            fix stim1 delay stim2 delay cue test 
        """
        # par['sample_time_rng'] = range((par['dead_time']+par['fix_time'])//par['dt'], (par['dead_time']+par['fix_time']+par['sample_time'])//par['dt'])
        sample_time_rng = []
        test_time_rng = []
        mask_time_rng = []

        for n in range(2):
            sample_time_rng.append(range((par['dead_time']+par['fix_time']+n*par['sample_time']+n*par['delay_time'])//par['dt'], \
                (par['dead_time']+par['fix_time']+(n+1)*par['sample_time']+n*par['delay_time'])//par['dt']))

        test_time_rng.append(range((par['dead_time']+par['fix_time']+2*par['sample_time']+2*par['delay_time']+par['delay_time']//2)//par['dt'], \
            (par['dead_time']+par['fix_time']+2*par['sample_time']+2*par['delay_time']+par['delay_time']//2+par['test_time'])//par['dt']))
        mask_time_rng.append(range((par['dead_time']+par['fix_time']+2*par['sample_time']+2*par['delay_time']+par['delay_time']//2)//par['dt'], \
            (par['dead_time']+par['fix_time']+2*par['sample_time']+2*par['delay_time']+par['delay_time']//2+par['mask_duration'])//par['dt']))
            

        fix_time_rng = []
        fix_time_rng.append(range(par['dead_time']//par['dt'], (par['dead_time']+par['fix_time'])//par['dt']))
        fix_time_rng.append(range((par['dead_time']+par['fix_time']+par['sample_time'])//par['dt'], \
            (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']))
        fix_time_rng.append(range((par['dead_time']+par['fix_time']+2*par['sample_time']+par['delay_time'])//par['dt'], \
            (par['dead_time']+par['fix_time']+2*par['sample_time']+2*par['delay_time'])//par['dt']))


        # duration of mask after test onset
        mask_duration = par['mask_duration']//par['dt']


        trial_info = {'desired_output'  :  np.zeros((par['num_time_steps'], par['batch_size'], par['n_output']),dtype=np.float32),
                      'train_mask'      :  np.ones((par['num_time_steps'], par['batch_size']),dtype=np.float32),
                      'sample'          :  np.zeros((par['batch_size'],2),dtype=np.int8),
                      'test'            :  np.zeros((par['batch_size']),dtype=np.int8),
                      'rule'            :  np.zeros((par['batch_size']),dtype=np.int8),
                      'match'           :  np.zeros((par['batch_size']),dtype=np.int8),
                      'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(par['num_time_steps'], par['batch_size'], par['n_input']))}


        for t in range(par['batch_size']):

            # generate sample, match, rule and prob params
            for i in range(2):
                trial_info['sample'][t,i] = np.random.randint(par['num_motion_dirs'])
                

            # determine test stimulu based on sample and match status
            trial_info['match'][t] = np.random.randint(2)
            trial_info['rule'][t] = np.random.randint(2)
            cue_stim = trial_info['rule'][t]
            if test_mode:
                trial_info['test'][t] = np.random.randint(par['num_motion_dirs'])
            else:
                if trial_info['match'][t]== 1:
                    trial_info['test'][t] = trial_info['sample'][t,trial_info['rule'][t]]
                else:
                    bad_directions = [trial_info['sample'][t,trial_info['rule'][t]]]
                    possible_stim = np.setdiff1d(list(range(par['num_motion_dirs'])), bad_directions)
                    trial_info['test'][t] = possible_stim[np.random.randint(len(possible_stim))]

            

            """
            Calculate input neural activity based on trial params
            """
            # SAMPLE stimuli
            trial_info['neural_input'][sample_time_rng[0], t, :] += \
                np.reshape(self.motion_tuning[:,0,trial_info['sample'][t,0]],(1,-1))
            trial_info['neural_input'][sample_time_rng[1], t, :] += \
                np.reshape(self.motion_tuning[:,0,trial_info['sample'][t,1]],(1,-1))

            # TEST stimuli
            trial_info['neural_input'][test_time_rng[0], t, :] += \
                np.reshape(self.motion_tuning[:,0,trial_info['test'][t]],(1,-1))

            # FIXATION
            trial_info['neural_input'][fix_time_rng[0], t, :] += np.reshape(self.fix_tuning[:,0],(1,-1))
            trial_info['neural_input'][fix_time_rng[1], t, :] += np.reshape(self.fix_tuning[:,0],(1,-1))
            trial_info['neural_input'][fix_time_rng[2], t, :] += np.reshape(self.fix_tuning[:,0],(1,-1))

            # RULE CUE
            trial_info['neural_input'][par['rule_time_rng'][0], t, :] += np.reshape(self.rule_tuning[:,trial_info['rule'][t]],(1,-1))
            # PROBE
            # increase reponse of all stim tuned neurons by 10
            """
            if trial_info['probe'][t,0]:
                trial_info['neural_input'][:est,probe_time1,t] += 10
            if trial_info['probe'][t,1]:
                trial_info['neural_input'][:est,probe_time2,t] += 10
            """

            """
            Desired outputs
            """
            # FIXATION
            trial_info['desired_output'][fix_time_rng[0], t, 0] = 1
            trial_info['desired_output'][fix_time_rng[1], t, 0] = 1
            trial_info['desired_output'][fix_time_rng[2], t, 0] = 1
            # TEST
            trial_info['train_mask'][ test_time_rng[0], t] *= par['test_cost_multiplier'] # can use a greater weight for test period if needed
            if trial_info['match'][t] == 1:
                trial_info['desired_output'][test_time_rng[0], t, 2] = 1
            else:
                trial_info['desired_output'][test_time_rng[0], t, 1] = 1

            # set to mask equal to zero during the dead time, and during the first times of test stimuli
            trial_info['train_mask'][:par['dead_time']//par['dt'], t] = 0
            trial_info['train_mask'][mask_time_rng[0], t] = 0

        return trial_info

    def create_tuning_functions(self):

        """
        Generate tuning functions for the Postle task
        """

        motion_tuning = np.zeros((par['n_input'], par['num_receptive_fields'], par['num_motion_dirs']))
        fix_tuning = np.zeros((par['n_input'], 1))
        rule_tuning = np.zeros((par['n_input'], par['num_rules']))


        # generate list of prefered directions
        # dividing neurons by 2 since two equal groups representing two modalities
        pref_dirs = np.float32(np.arange(0,360,360/(par['num_motion_tuned']//par['num_receptive_fields'])))

        # generate list of possible stimulus directions
        stim_dirs = np.float32(np.arange(0,360,360/par['num_motion_dirs']))

        for n in range(par['num_motion_tuned']//par['num_receptive_fields']):
            for i in range(par['num_motion_dirs']):
                for r in range(par['num_receptive_fields']):
                    if par['trial_type'] == 'distractor':
                        if n%par['num_motion_dirs'] == i:
                            motion_tuning[n,0,i] = par['tuning_height']
                    else:
                        d = np.cos((stim_dirs[i] - pref_dirs[n])/180*np.pi)
                        n_ind = n+r*par['num_motion_tuned']//par['num_receptive_fields']
                        motion_tuning[n_ind,r,i] = par['tuning_height']*np.exp(par['kappa']*d)/np.exp(par['kappa'])

        for n in range(par['num_fix_tuned']):
            fix_tuning[par['num_motion_tuned']+n,0] = par['tuning_height']

        for n in range(par['num_rule_tuned']):
            for i in range(par['num_rules']):
                if n%par['num_rules'] == i:
                    rule_tuning[par['num_motion_tuned']+par['num_fix_tuned']+n,i] = par['tuning_height']*par['rule_cue_multiplier']


        return motion_tuning, fix_tuning, rule_tuning