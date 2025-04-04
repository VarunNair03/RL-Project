from utils.models import DQN, FeatureExtractor
from utils.tools import ReplayMemory, Transition, eval_stats_at_threshold, extract, show_new_bdbox
import os
import imageio
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from tqdm.notebook import tqdm
from config import SAVE_MODEL_PATH, use_cuda, transform, criterion



class Agent():
    def __init__(self, classe, alpha=0.2, nu=3.0, threshold=0.5, num_episodes=15, load=False, arch='resnet50'):
        self.BATCH_SIZE = 100
        self.GAMMA = 0.900
        self.EPS = 1
        self.TARGET_UPDATE = 1
        self.save_path = SAVE_MODEL_PATH
        screen_height, screen_width = 224, 224
        self.n_actions = 9
        self.classe = classe

        self.feature_extractor = FeatureExtractor(arch=arch)
        
        self.feature_dim = self.feature_extractor.get_feature_dim()
        print(self.feature_dim)
        
        if use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        
        if not load:
            self.policy_net = DQN(screen_height, screen_width, self.n_actions, self.feature_dim)
        else:
            self.policy_net = self.load_network()
            
        self.target_net = DQN(screen_height, screen_width, self.n_actions, self.feature_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.feature_extractor.eval()
        
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.target_net = self.target_net.to(self.device)
        self.policy_net = self.policy_net.to(self.device)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=1e-6)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episode_durations = []
        
        self.alpha = alpha # €[0, 1]  Scaling factor
        self.nu = nu # Reward of Trigger
        self.threshold = threshold
        self.actions_history = []
        self.num_episodes = num_episodes
        self.actions_history += [[100]*9]*20

    def save_network(self):

        torch.save(self.policy_net, self.save_path+"_"+self.classe)
        print('Saved')

    def load_network(self):

        if not use_cuda:
            return torch.load(self.save_path+"_"+self.classe, map_location=torch.device('cpu'))
        return torch.load(self.save_path+"_"+self.classe)



    def intersection_over_union(self, box1, box2, epsilon=1e-8):

        x11, x21, y11, y21 = box1
        x12, x22, y12, y22 = box2
        
        yi1 = max(y11, y12)
        xi1 = max(x11, x12)
        yi2 = min(y21, y22)
        xi2 = min(x21, x22)
        inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
        box1_area = (x21 - x11) * (y21 - y11)
        box2_area = (x22 - x12) * (y22 - y12)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / (union_area + epsilon)
        return iou

    def compute_reward(self, actual_state, previous_state, ground_truth):
        
        res = self.intersection_over_union(actual_state, ground_truth) - self.intersection_over_union(previous_state, ground_truth)
        if res <= 0:
            return -1
        return 1
      
    def rewrap(self, coord):
        return min(max(coord,0), 224)
      
    def compute_trigger_reward(self, actual_state, ground_truth):
        res = self.intersection_over_union(actual_state, ground_truth)
        if res>=self.threshold:
            return self.nu
        return -1*self.nu

    def get_best_next_action(self, actions, ground_truth):
        
        positive_actions = []
        negative_actions = []
        actual_equivalent_coord = self.calculate_position_box(actions)
        for i in range(0, 9):
            copy_actions = actions + [i]  # More efficient than copy().append(i)
            new_equivalent_coord = self.calculate_position_box(copy_actions)
            if i!=0:
                reward = self.compute_reward(new_equivalent_coord, actual_equivalent_coord, ground_truth)
            else:
                reward = self.compute_trigger_reward(new_equivalent_coord,  ground_truth)
            
            if reward>=0:
                positive_actions.append(i)
            else:
                negative_actions.append(i)
        if len(positive_actions)==0:
            return random.choice(negative_actions)
        return random.choice(positive_actions)


    def select_action(self, state, actions, ground_truth):

        sample = random.random()
        eps_threshold = self.EPS
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                inpu = state.to(self.device)
                qval = self.policy_net(inpu)
                _, predicted = torch.max(qval.data,1)
                action = predicted[0] # + 1
                try:
                  return action.cpu().numpy()[0]
                except:
                  return action.cpu().numpy()
        else:
            return self.get_best_next_action(actions, ground_truth) 

    def select_action_model(self, state):
        with torch.no_grad():
                if use_cuda:
                    inpu = state.cuda()
                else:
                    inpu = state
                qval = self.policy_net(inpu)
                _, predicted = torch.max(qval.data,1)
                #print("Predicted : "+str(qval.data))
                action = predicted[0] # + 1
                #print(action)
                return action

        
    def optimize_model(self):
        """
        Performs a single optimization step for the DQN.
        """
        # Return if there are not enough samples in memory
        if len(self.memory) < self.BATCH_SIZE:
            return

        # Sample random batch of transitions from memory
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))  # Unpack batch

        # Mask of non-final states
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool, device=self.device)

        # Extract next states and stack them
        next_states = [s for s in batch.next_state if s is not None]
        if next_states:
            with torch.no_grad():
                non_final_next_states = torch.cat(next_states).to(self.device)
        else:
            non_final_next_states = None  # No valid next states

        # Convert batch data to tensors
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).view(-1, 1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float, device=self.device).view(-1, 1)

        # Compute Q(s_t, a) - get the Q-values for the taken actions
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for non-final next states
        next_state_values = torch.zeros(self.BATCH_SIZE, 1, device=self.device)

        if non_final_next_states is not None:
            with torch.no_grad():
                d = self.target_net(non_final_next_states)
                next_state_values[non_final_mask] = d.max(1)[0].view(-1, 1)

        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute loss
        loss = criterion(state_action_values, expected_state_action_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    
    def compose_state(self, image, dtype=torch.float32):
        image_feature = self.get_features(image, dtype)
        image_feature = image_feature.view(1,-1)
        #print("image feature : "+str(image_feature.shape))
        history_flatten = torch.tensor(self.actions_history, dtype=dtype, device=self.device).view(1, -1)
        state = torch.cat((image_feature, history_flatten), 1)
        return state
    
    def get_features(self, image, dtype=torch.float32):
        # Convert image to tensor if needed
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=dtype)
        elif isinstance(image, Image.Image):  # If it's a PIL image
            image = transform.ToTensor()(image)

        image = image.unsqueeze(0)  # Add batch dimension

        # Move image to the correct device
        image = image.to(self.device)

        # Extract features without tracking gradients
        with torch.no_grad():
            feature = self.feature_extractor(image)

        return feature

    
    def update_history(self, action):
        action_vector = torch.zeros(9)
        action_vector[action] = 1
        size_history_vector = len(torch.nonzero(self.actions_history))
        if size_history_vector < 9:
            self.actions_history[size_history_vector][action] = 1
        else:
            for i in range(8,0,-1):
                self.actions_history[i][:] = self.actions_history[i-1][:]
            self.actions_history[0][:] = action_vector[:] 
        return self.actions_history

    def calculate_position_box(self, actions, xmin=0, xmax=224, ymin=0, ymax=224):

        
        alpha_h = self.alpha * (  ymax - ymin )
        alpha_w = self.alpha * (  xmax - xmin )
        real_x_min, real_x_max, real_y_min, real_y_max = 0, 224, 0, 224

        for r in actions:
            if r == 1: # Right
                real_x_min += alpha_w
                real_x_max += alpha_w
            if r == 2: # Left
                real_x_min -= alpha_w
                real_x_max -= alpha_w
            if r == 3: # Up 
                real_y_min -= alpha_h
                real_y_max -= alpha_h
            if r == 4: # Down
                real_y_min += alpha_h
                real_y_max += alpha_h
            if r == 5: # Bigger
                real_y_min -= alpha_h
                real_y_max += alpha_h
                real_x_min -= alpha_w
                real_x_max += alpha_w
            if r == 6: # Smaller
                real_y_min += alpha_h
                real_y_max -= alpha_h
                real_x_min += alpha_w
                real_x_max -= alpha_w
            if r == 7: # Fatter
                real_y_min += alpha_h
                real_y_max -= alpha_h
            if r == 8: # Taller
                real_x_min += alpha_w
                real_x_max -= alpha_w
        real_x_min, real_x_max, real_y_min, real_y_max = self.rewrap(real_x_min), self.rewrap(real_x_max), self.rewrap(real_y_min), self.rewrap(real_y_max)
        return [real_x_min, real_x_max, real_y_min, real_y_max]

    def get_max_bdbox(self, ground_truth_boxes, actual_coordinates ):
        max_iou = -1.0  # Initialize IoU with a very low value
        max_gt = None   # Set to None instead of an empty list

        for gt in ground_truth_boxes:
            iou = self.intersection_over_union(actual_coordinates, gt)
            if iou > max_iou:
                max_iou = iou
                max_gt = gt

        return max_gt

    def predict_image(self, image, plot=False):

        self.policy_net.eval()

        done = False
        all_actions = []
        self.actions_history = torch.ones((9,9))
        state = self.compose_state(image)
        original_image = image.clone()
        new_image = image

        steps = 0
        
        while not done:
            steps += 1
            action = self.select_action_model(state)
            all_actions.append(action)
            if action == 0:
                next_state = None
                new_equivalent_coord = self.calculate_position_box(all_actions)
                done = True
            else:
                # Mise à jour de l'historique
                self.actions_history = self.update_history(action)
                new_equivalent_coord = self.calculate_position_box(all_actions)            
                
                # Récupération du contenu de la boite englobante
                new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]), int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]
                try:
                    new_image = transform(new_image)
                except ValueError:
                    break            
                
                # Composition : état + historique des 9 dernières actions
                next_state = self.compose_state(new_image)
            
            if steps == 40:
                done = True
            
            # Déplacement au nouvel état
            state = next_state
            image = new_image
        
            if plot:
                show_new_bdbox(original_image, new_equivalent_coord, color='b', count=steps)
        

        # Génération d'un GIF représentant l'évolution de la prédiction
        if plot:
            #images = []
            tested = 0
            while os.path.isfile('media/movie_'+str(tested)+'.gif'):
                tested += 1
            # filepaths
            fp_out = "media/movie_"+str(tested)+".gif"
            images = []
            for count in range(1, steps+1):
                images.append(imageio.imread(str(count)+".png"))
            
            imageio.mimsave(fp_out, images)
            
            for count in range(1, steps):
                os.remove(str(count)+".png")
        return new_equivalent_coord


    
    def evaluate(self, dataset):
        """
            Evaluation des performances du model sur un jeu de données.
            Entrée :
                - Jeu de données de test.
            Sortie :
                - Statistiques d'AP et RECALL.

        """
        ground_truth_boxes = []
        predicted_boxes = []
        print("Predicting boxes...")
        for key, value in dataset.items():
            image, gt_boxes = extract(key, dataset)
            bbox = self.predict_image(image)
            ground_truth_boxes.append(gt_boxes)
            predicted_boxes.append(bbox)
        print("Computing recall and ap...")
        stats = eval_stats_at_threshold(predicted_boxes, ground_truth_boxes)
        print("Final result : \n"+str(stats))
        return stats

    def train(self, train_loader):

        xmin = 0.0
        xmax = 224.0
        ymin = 0.0
        ymax = 224.0

        for i_episode in range(self.num_episodes):
            print("Episode "+str(i_episode))
            for key, value in  train_loader.items():
                image, ground_truth_boxes = extract(key, train_loader)
                original_image = image.clone()
                ground_truth = ground_truth_boxes[0]
                all_actions = []
        
                # Initialize the environment and state
                self.actions_history = torch.ones((9,9))
                state = self.compose_state(image)
                original_coordinates = [xmin, xmax, ymin, ymax]
                new_image = image
                done = False
                t = 0
                actual_equivalent_coord = original_coordinates
                new_equivalent_coord = original_coordinates
                while not done:
                    t += 1
                    action = self.select_action(state, all_actions, ground_truth)
                    all_actions.append(action)
                    if action == 0:
                        next_state = None
                        new_equivalent_coord = self.calculate_position_box(all_actions)
                        closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord )
                        reward = self.compute_trigger_reward(new_equivalent_coord,  closest_gt)
                        done = True

                    else:
                        self.actions_history = self.update_history(action)
                        new_equivalent_coord = self.calculate_position_box(all_actions)
                        
                        new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]), int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]
                        try:
                            new_image = transform(new_image)
                        except ValueError:
                            break                        

                        next_state = self.compose_state(new_image)
                        closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord )
                        reward = self.compute_reward(new_equivalent_coord, actual_equivalent_coord, closest_gt)
                        
                        actual_equivalent_coord = new_equivalent_coord
                    if t == 20:
                        done = True
                    self.memory.push(state, int(action), next_state, reward)

                    # Move to the next state
                    state = next_state
                    image = new_image
                    # Perform one step of the optimization (on the target network)
                    self.optimize_model()
                    
            
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if i_episode<5:
                self.EPS -= 0.18
            self.save_network()

            print('Complete')

    def train_validate(self, train_loader, valid_loader):

        op = open("logs_over_epochs", "w")
        op.write("NU = "+str(self.nu))
        op.write("ALPHA = "+str(self.alpha))
        op.write("THRESHOLD = "+str(self.threshold))
        xmin = 0.0
        xmax = 224.0
        ymin = 0.0
        ymax = 224.0
        for i_episode in range(self.num_episodes):  
            print("Episode "+str(i_episode))
            for key, value in  train_loader.items():
                image, ground_truth_boxes = extract(key, train_loader)
                original_image = image.clone()
                ground_truth = ground_truth_boxes[0]
                all_actions = []
        
                # Initialize the environment and state
                self.actions_history = torch.ones((9,9))
                state = self.compose_state(image)
                original_coordinates = [xmin, xmax, ymin, ymax]
                new_image = image
                done = False
                t = 0
                actual_equivalent_coord = original_coordinates
                new_equivalent_coord = original_coordinates
                while not done:
                    t += 1
                    action = self.select_action(state, all_actions, ground_truth)
                    all_actions.append(action)
                    if action == 0:
                        next_state = None
                        new_equivalent_coord = self.calculate_position_box(all_actions)
                        closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord )
                        reward = self.compute_trigger_reward(new_equivalent_coord,  closest_gt)
                        done = True

                    else:
                        self.actions_history = self.update_history(action)
                        new_equivalent_coord = self.calculate_position_box(all_actions)
                        
                        new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]), int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]
                        try:
                            new_image = transform(new_image)
                        except ValueError:
                            break                        
                        if False:
                            show_new_bdbox(original_image, ground_truth, color='r')
                            show_new_bdbox(original_image, new_equivalent_coord, color='b')
                            

                        next_state = self.compose_state(new_image)
                        closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord )
                        reward = self.compute_reward(new_equivalent_coord, actual_equivalent_coord, closest_gt)
                        
                        actual_equivalent_coord = new_equivalent_coord
                    if t == 20:
                        done = True
                    self.memory.push(state, int(action), next_state, reward)

                    # Vers le nouvel état
                    state = next_state
                    image = new_image
                    # Optimisation
                    self.optimize_model()
                    
            stats = self.evaluate(valid_loader)
            op.write("\n")
            op.write("Episode "+str(i_episode))
            op.write(str(stats))
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if i_episode<5:
                self.EPS -= 0.18
            self.save_network()
            
            print('Complete')