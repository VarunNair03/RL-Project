#import torchvision.datasets.SBDataset as sbd
from utils.models import *
from utils.tools import *
import os
import imageio
import math
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets

from itertools import count
from PIL import Image
import torch.optim as optim
import cv2 as cv
from torch.autograd import Variable

from tqdm.notebook import tqdm
from config import *

import glob
from PIL import Image

class Agent():
    def __init__(self, classe, alpha=0.2, nu=3.0, threshold=0.5, num_episodes=15, load=False):
        """
            Classe initialisant l'ensemble des paramètres de l'apprentissage, un agent est associé à une classe donnée du jeu de données.
        """
        self.BATCH_SIZE = 100
        self.GAMMA = 0.900
        self.EPS = 1
        self.TARGET_UPDATE = 1
        self.save_path = SAVE_MODEL_PATH
        screen_height, screen_width = 224, 224
        self.n_actions = 9
        self.classe = classe

        self.feature_extractor = FeatureExtractor()
        if not load:
            self.policy_net = DQN(screen_height, screen_width, self.n_actions)
        else:
            self.policy_net = self.load_network()
            
        self.target_net = DQN(screen_height, screen_width, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.feature_extractor.eval()
        if use_cuda:
          self.feature_extractor = self.feature_extractor.cuda()
          self.target_net = self.target_net.cuda()
          self.policy_net = self.policy_net.cuda()
        
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
        """
            Fonction de sauvegarde du Q-Network
        """
        torch.save(self.policy_net, self.save_path+"_"+self.classe)
        print('Saved')

    def load_network(self):
        """
            Récupération d'un Q-Network existant
        """
        if not use_cuda:
            return torch.load(self.save_path+"_"+self.classe, map_location=torch.device('cpu'))
        return torch.load(self.save_path+"_"+self.classe)



    def intersection_over_union(self, box1, box2, epsilon=1e-8):
        """
        Computes the Intersection over Union (IoU) metric.
        
        Args:
            box1 (list): Bounding box coordinates [x_min, x_max, y_min, y_max] for prediction.
            box2 (list): Bounding box coordinates [x_min, x_max, y_min, y_max] for ground truth.
            epsilon (float): Small value to prevent division by zero.

        Returns:
            float: IoU score.
        """
        x11, x21, y11, y21 = box1
        x12, x22, y12, y22 = box2

        # Compute intersection
        xi1, yi1 = max(x11, x12), max(y11, y12)
        xi2, yi2 = min(x21, x22), min(y21, y22)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Compute union
        box1_area = (x21 - x11) * (y21 - y11)
        box2_area = (x22 - x12) * (y22 - y12)
        union_area = box1_area + box2_area - inter_area

        return inter_area / (union_area + epsilon)


    def compute_reward(self, actual_state, previous_state, ground_truth):
        """
        Computes the reward for non-terminal states based on IoU difference.
        
        Args:
            actual_state (list): Current bounding box [x_min, x_max, y_min, y_max].
            previous_state (list): Previous bounding box [x_min, x_max, y_min, y_max].
            ground_truth (list): Ground truth bounding box [x_min, x_max, y_min, y_max].

        Returns:
            int: Reward (-1 for decrease/no change, +1 for improvement).
        """
        iou_diff = self.intersection_over_union(actual_state, ground_truth) - \
                self.intersection_over_union(previous_state, ground_truth)
        return 1 if iou_diff > 0 else -1


    def rewrap(self, coord):
        """
        Clamps coordinate values within valid image bounds [0, 224].
        
        Args:
            coord (float): Coordinate value.

        Returns:
            float: Clamped coordinate.
        """
        return max(0, min(coord, 224))


    def compute_trigger_reward(self, actual_state, ground_truth):
        """
        Computes the reward for terminal states when the trigger action is taken.
        
        Args:
            actual_state (list): Current bounding box [x_min, x_max, y_min, y_max].
            ground_truth (list): Ground truth bounding box [x_min, x_max, y_min, y_max].

        Returns:
            float: Reward (self.nu if IoU >= threshold, -self.nu otherwise).
        """
        return self.nu if self.intersection_over_union(actual_state, ground_truth) >= self.threshold else -self.nu

    def get_best_next_action(self, actions, ground_truth):
        """
        Determines the best next action based on the current state and ground truth.
        
        Args:
            actions (list): List of executed actions.
            ground_truth (list): Ground truth bounding box.

        Returns:
            int: Index of the best possible action.
        """
        actual_box = self.calculate_position_box(actions)
        positive_actions, negative_actions = [], []

        for i in range(9):
            new_actions = actions + [i]  # More efficient than list copy
            new_box = self.calculate_position_box(new_actions)

            reward = self.compute_reward(new_box, actual_box, ground_truth) if i != 0 else \
                    self.compute_trigger_reward(new_box, ground_truth)

            (positive_actions if reward >= 0 else negative_actions).append(i)

        return random.choice(positive_actions) if positive_actions else random.choice(negative_actions)


    def select_action(self, state, actions, ground_truth):
        """
        Selects an action based on the current state.

        Args:
            state (torch.Tensor): Current state.
            actions (list): List of executed actions.
            ground_truth (list): Ground truth bounding box.

        Returns:
            int: Selected action.
        """
        self.steps_done += 1

        if random.random() > self.EPS:  # Exploitation: Choose the best action from policy network
            with torch.no_grad():
                inpu = Variable(state).cuda() if use_cuda else Variable(state)
                action = self.policy_net(inpu).argmax(dim=1).cpu().numpy()
                return action.item() if isinstance(action, np.ndarray) else action
        else:  # Exploration: Use expert agent for best next action
            return self.get_best_next_action(actions, ground_truth)


    def select_action_model(self, state):
        """
        Selects an action using the policy network based on the current state.

        Args:
            state (torch.Tensor): Current state (feature vector + action history).

        Returns:
            int: Selected action.
        """
        with torch.no_grad():
            state = state.cuda() if use_cuda else state
            action = self.policy_net(state).argmax(dim=1).cpu().item()
            return action


    def optimize_model(self):
        """
        Updates the policy network by sampling memory, computing loss, and performing backpropagation.
        """
        if len(self.memory) < self.BATCH_SIZE:
            return  # Insufficient memory to train

        # Sample a batch of transitions from replay memory
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Extract elements from the batch
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.LongTensor(batch.action).view(-1, 1).to(device)
        reward_batch = torch.FloatTensor(batch.reward).view(-1, 1).to(device)

        # Process next states
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool, device=device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

        # Compute Q(s_t, a) - model output for taken actions
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) using the target network
        next_state_values = torch.zeros(self.BATCH_SIZE, 1, device=device)
        if non_final_next_states.size(0) > 0:  # Ensure non-empty next states
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1, keepdim=True)[0]

        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute loss and backpropagate
        loss = criterion(state_action_values, expected_state_action_values.detach())  # Detach to avoid gradients through target
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def compose_state(self, image, dtype=torch.FloatTensor):
        """
        Composes the state representation by concatenating the feature vector and action history.

        Args:
            image (torch.Tensor): Input image.
            dtype (torch.dtype): Data type for processing (default: FloatTensor).

        Returns:
            torch.Tensor: Concatenated state representation.
        """
        image_feature = self.get_features(image, dtype).view(1, -1)
        history_flatten = self.actions_history.flatten().unsqueeze(0).to(dtype)
        return torch.cat((image_feature, history_flatten), dim=1)


    def get_features(self, image, dtype=torch.FloatTensor):
        """
        Extracts the feature vector from an image using the feature extractor.

        Args:
            image (torch.Tensor): Input image.
            dtype (torch.dtype): Data type for processing (default: FloatTensor).

        Returns:
            torch.Tensor: Extracted feature vector.
        """
        image = image.unsqueeze(0).to(dtype)
        if use_cuda:
            image = image.cuda()
        with torch.no_grad():  # Avoids computing gradients for inference
            return self.feature_extractor(image)


    def update_history(self, action):
        """
        Updates the action history by shifting past actions and adding the latest action.

        Args:
            action (int): Index of the last executed action.

        Returns:
            torch.Tensor: Updated action history.
        """
        action_vector = torch.zeros(9, device=self.actions_history.device)
        action_vector[action] = 1

        # Shift action history
        self.actions_history[1:] = self.actions_history[:-1].clone()
        self.actions_history[0] = action_vector
        return self.actions_history


    def calculate_position_box(self, actions, xmin=0, xmax=224, ymin=0, ymax=224):
        """
        Computes the final bounding box coordinates based on a sequence of actions.

        Args:
            actions (list): List of selected actions from the beginning.
            xmin (int): Minimum x-bound of the initial bounding box.
            xmax (int): Maximum x-bound of the initial bounding box.
            ymin (int): Minimum y-bound of the initial bounding box.
            ymax (int): Maximum y-bound of the initial bounding box.

        Returns:
            list: Final bounding box coordinates [x_min, x_max, y_min, y_max].
        """
        # Compute step sizes
        alpha_h = self.alpha * (ymax - ymin)
        alpha_w = self.alpha * (xmax - xmin)

        # Initialize bounding box coordinates
        x_min, x_max, y_min, y_max = 0, 224, 0, 224

        # Apply transformations based on actions
        for action in actions:
            if action == 1:  # Move Right
                x_min += alpha_w
                x_max += alpha_w
            elif action == 2:  # Move Left
                x_min -= alpha_w
                x_max -= alpha_w
            elif action == 3:  # Move Up
                y_min -= alpha_h
                y_max -= alpha_h
            elif action == 4:  # Move Down
                y_min += alpha_h
                y_max += alpha_h
            elif action == 5:  # Expand
                x_min -= alpha_w
                x_max += alpha_w
                y_min -= alpha_h
                y_max += alpha_h
            elif action == 6:  # Shrink
                x_min += alpha_w
                x_max -= alpha_w
                y_min += alpha_h
                y_max -= alpha_h
            elif action == 7:  # Make Fatter (reduce height)
                y_min += alpha_h
                y_max -= alpha_h
            elif action == 8:  # Make Taller (reduce width)
                x_min += alpha_w
                x_max -= alpha_w

        # Ensure bounding box remains within valid limits
        x_min, x_max, y_min, y_max = (
            self.rewrap(x_min), self.rewrap(x_max),
            self.rewrap(y_min), self.rewrap(y_max)
        )

        return [x_min, x_max, y_min, y_max]


    def get_max_bdbox(self, ground_truth_boxes, actual_coordinates):
        """
        Finds the ground truth bounding box that has the highest IoU with the current state.

        Args:
            ground_truth_boxes (list): List of ground truth bounding boxes.
            actual_coordinates (list): Current bounding box coordinates.

        Returns:
            list: Ground truth bounding box with the highest IoU.
        """
        if not ground_truth_boxes:
            return []

        max_iou = 0
        best_gt = None

        for gt in ground_truth_boxes:
            iou = self.intersection_over_union(actual_coordinates, gt)
            if iou > max_iou:
                max_iou = iou
                best_gt = gt

        return best_gt if best_gt is not None else []


    def predict_image(self, image, plot=False):
        """
            Prédit la boite englobante d'une image
            Entrée :
                - Image redimensionnée.
            Sortie :
                - Coordonnées boite englobante.
        """

        # Passage du Q-Network en mode évaluation
        self.policy_net.eval()
        xmin = 0
        xmax = 224
        ymin = 0
        ymax = 224

        done = False
        all_actions = []
        self.actions_history = torch.ones((9,9))
        state = self.compose_state(image)
        original_image = image.clone()
        new_image = image

        steps = 0
        
        # Tant que le trigger n'est pas déclenché ou qu'on a pas atteint les 40 steps
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
        """
            Fonction d'entraînement du modèle.
            Entrée :
                - Jeu de données d'entraînement.
        """
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
        """
            Entraînement du modèle et à chaque épisode test de l'efficacité sur le jeu de test et sauvegarde des résultats dans un fichier de logs.
        """
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