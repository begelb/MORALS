import torch
import os
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm
from MORALS.models import *
from .topological_autoencoders.approx_based import TopologicalSignatureDistance

class TrainingConfig:
    def __init__(self, weights_str):
        self.weights_str = weights_str
        self.parse_config()
    
    def parse_config(self):
        ids = self.weights_str.split('_')
        self.weights = []
        for _, id in enumerate(ids):
            self.weights.append([float(e) for e in id.split('x')[:-1]])
            if len(self.weights[-1]) != 5:
                print("Expected 5 values per training config, got ", len(self.weights[-1]))
                raise ValueError
    
    def __getitem__(self, key):
        return self.weights[key]
    
    def __len__(self):
        return len(self.weights)

class LabelsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(LabelsLoss, self).__init__()
        self.reduction = reduction
        self.scale = 100.0

    def forward(self, x, y):
        pairwise_distance = torch.linalg.vector_norm(x - y, ord=2, dim=1)
        loss = torch.sigmoid(-self.scale * pairwise_distance)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError("Invalid reduction type")

class Training:
    def __init__(self, config, loaders, verbose):
        self.encoder = Encoder(config)
        self.dynamics = LatentDynamics(config)
        self.decoder = Decoder(config)
        self.latent_norm = torch.nn.Parameter(data=torch.ones(1),
                                              requires_grad=True) # to do: understand why this is used
        self.topo_sig = TopologicalSignatureDistance()

        self.verbose = bool(verbose)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        self.encoder.to(self.device)
        self.dynamics.to(self.device)
        self.decoder.to(self.device)

        self.dynamics_train_loader = loaders['train_dynamics']
        self.dynamics_test_loader = loaders['test_dynamics']
        self.labels_train_loader = loaders['train_labels']
        self.labels_test_loader = loaders['test_labels']

        self.reset_losses()

        self.dynamics_criterion = nn.MSELoss(reduction='mean')
        self.labels_criterion = LabelsLoss(reduction='mean')

        self.lr = config["learning_rate"]

        self.model_dir = config["model_dir"]
        self.log_dir = config["log_dir"]

    def save_models(self, subfolder='', suffix=''):
        save_path = os.path.join(self.model_dir, subfolder)
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.encoder, os.path.join(save_path, 'encoder' + suffix + '.pt'))
        torch.save(self.dynamics, os.path.join(save_path, 'dynamics' + suffix + '.pt'))
        torch.save(self.decoder, os.path.join(save_path, 'decoder' + suffix + '.pt'))
    
    def load_models(self):
        self.encoder = torch.load(os.path.join(self.model_dir, 'encoder_.pt'))
        self.dynamics = torch.load(os.path.join(self.model_dir, 'dynamics_.pt'))
        self.decoder = torch.load(os.path.join(self.model_dir, 'decoder_.pt'))
    
    def save_logs(self, suffix):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        with open(os.path.join(self.log_dir, 'train_losses_' + suffix + '.pkl'), 'wb') as f:
            pickle.dump(self.train_losses, f)
        
        with open(os.path.join(self.log_dir, 'test_losses_' + suffix + '.pkl'), 'wb') as f:
            pickle.dump(self.test_losses, f)
    
    def reset_losses(self):
        self.train_losses = {'loss_ae1': [], 'loss_ae2': [], 'loss_dyn': [], 'loss_contrastive': [], 'loss_topo': [], 'loss_total': []}
        self.test_losses = {'loss_ae1': [], 'loss_ae2': [], 'loss_dyn': [], 'loss_contrastive': [], 'loss_topo': [], 'loss_total': []}
    
    def forward(self, x_t, x_tau):
        x_t = x_t.to(self.device)
        x_tau = x_tau.to(self.device)

        # z_t = E(x_t)
        z_t = self.encoder(x_t)

        # x_t_pred = D(E(x_t))
        x_t_pred = self.decoder(z_t)

        # z_tau = E(x_tau)
        z_tau = self.encoder(x_tau)

        # x_tau_pred = D(E(x_tau))
        # this variable does not get passed forward?
        x_tau_pred = self.decoder(z_tau)

        # z_tau_pred = latent_dynamics(E(x_t))
        z_tau_pred = self.dynamics(z_t)

        # x_tau_pred_dyn = D(latent_dynamics(E(x_t)))
        x_tau_pred_dyn = self.decoder(z_tau_pred)

        return (x_t, x_tau, x_t_pred, z_tau, z_tau_pred, x_tau_pred_dyn)

    def dynamics_losses(self, forward_pass, weight):
        x_t, x_tau, x_t_pred, z_tau, z_tau_pred, x_tau_pred_dyn = forward_pass

        # x_t_pred = D(E(x_t)) so this is the reconstruction loss for x_t
        loss_ae1 = self.dynamics_criterion(x_t, x_t_pred)

        # x_tau_pred_dyn = D(latent_dynamics(E(x_t))) so this is also a dynamics loss. Should x_tau_pred_dyn be x_tau_pred?
        loss_ae2 = self.dynamics_criterion(x_tau, x_tau_pred_dyn)

        # z_tau_pred = latent_dynamics(E(x_t)) and z_tau = E(x_tau) so this is the dynamics loss 
        loss_dyn = self.dynamics_criterion(z_tau_pred, z_tau)
        loss_total = loss_ae1 * weight[0] + loss_ae2 * weight[1] + loss_dyn * weight[2]
        return loss_ae1, loss_ae2, loss_dyn, loss_total

    def labels_losses(self, encodings, pairs, weight):
        return self.labels_criterion(encodings[pairs['successes']], encodings[pairs['failures']]) * weight
    
    @staticmethod
    def _compute_distance_matrix(x, p=2):
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
        return distances
    
    def topological_losses(self, forward_pass, weight):
        x_t, x_tau, x_t_pred, z_tau, z_tau_pred, x_tau_pred_dyn = forward_pass

        x = torch.cat((x_t, x_tau), dim = 0)
        dimensions = x.size()

        x_distances = self._compute_distance_matrix(x)

        # normalize
        # to do: understand exception in the original code for 4-d
        x_distances = x_distances / x_distances.max()

        # to do: get latent from the forward pass, for now I am recomputing in order to not change the forward pass
        z_t = self.encoder(x_t)
        z_tau = self.encoder(x_tau)

        latent = torch.cat((z_t, z_tau), dim = 0)
        latent_distances = self._compute_distance_matrix(latent)
        latent_distances = latent_distances / self.latent_norm

        topo_error, topo_error_components = self.topo_sig(
            x_distances, latent_distances)
        
        # normalize topo_error according to batch_size
        batch_size = dimensions[0]
        topo_error = topo_error / float(batch_size)
        topo_error_weighted = topo_error * weight[4]
    
        return topo_error_weighted


    def train(self, epochs=1000, patience=50, weight=[1,1,1,0,1]):
        '''
        Function that trains all the models with all the losses and weight.
        It will stop if the test loss does not improve for "patience" epochs.
        '''
        print("Training with weights: ", weight)
        weight_bool = [bool(i) for i in weight]
        print('weight bool: ', weight_bool)

        list_parameters = (weight_bool[0] or weight_bool[1] or weight_bool[2]) * (list(self.encoder.parameters()) + list(self.decoder.parameters()))
        list_parameters += (weight_bool[1] or weight_bool[2]) * list(self.dynamics.parameters())
        optimizer = torch.optim.Adam(list_parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, patience=patience, verbose=True)
        for epoch in tqdm(range(epochs)):
            loss_ae1_train = 0
            loss_ae2_train = 0
            loss_dyn_train = 0
            loss_contrastive_train = 0
            loss_topo_train = 0

            epoch_train_loss = 0
            epoch_test_loss  = 0


            if weight_bool[0] or weight_bool[1] or weight_bool[2]: 
                # put encoder and decoder in train mode
                self.encoder.train() 
                self.decoder.train() 
            if weight_bool[1] or weight_bool[2]: 
                # put dynamics in train mode
                self.dynamics.train()

            num_batches = min(len(self.dynamics_train_loader), len(self.labels_train_loader))
            for (x_t, x_tau), (pairs, x_final) in zip(self.dynamics_train_loader, self.labels_train_loader):
                optimizer.zero_grad()

                # Forward pass (apply all models)
                forward_pass = self.forward(x_t, x_tau)
                # Compute losses
                loss_ae1, loss_ae2, loss_dyn, loss_total = self.dynamics_losses(forward_pass, weight)
                if weight[4] != 0:
                    loss_topo = self.topological_losses(forward_pass, weight)
                    loss_total += loss_topo
                    loss_topo_train += loss_topo.item()

                loss_con = 0
                if weight[3] != 0:
                    x_final = x_final.to(self.device)
                    z_final = self.encoder(x_final)
                    loss_con = self.labels_losses(z_final, pairs, weight[3])
                    loss_total += loss_con
                    loss_contrastive_train += loss_con.item()
                # Backward pass
                loss_total.backward()
                optimizer.step()

                loss_ae1_train += loss_ae1.item()
                loss_ae2_train += loss_ae2.item()
                loss_dyn_train += loss_dyn.item()

                epoch_train_loss += loss_total.item()

            epoch_train_loss /= num_batches

            self.train_losses['loss_ae1'].append(loss_ae1_train / num_batches)
            self.train_losses['loss_ae2'].append(loss_ae2_train / num_batches)
            self.train_losses['loss_dyn'].append(loss_dyn_train / num_batches)
            self.train_losses['loss_contrastive'].append(loss_contrastive_train / num_batches)
            self.train_losses['loss_topo'].append(loss_topo_train / num_batches)
            self.train_losses['loss_total'].append(epoch_train_loss)

            with torch.no_grad():
                loss_ae1_test = 0
                loss_ae2_test = 0
                loss_dyn_test = 0
                loss_contrastive_test = 0
                loss_topo_test = 0

                if weight_bool[0] or weight_bool[1] or weight_bool[2]:  
                    self.encoder.eval() 
                    self.decoder.eval() 
                if weight_bool[1] or weight_bool[2]: 
                    self.dynamics.eval()

                num_batches = min(len(self.dynamics_test_loader), len(self.labels_test_loader))
                for (x_t, x_tau), (pairs, x_final) in zip(self.dynamics_test_loader, self.labels_test_loader):
                    # Forward pass
                    forward_pass = self.forward(x_t, x_tau)
                    # Compute losses
                    loss_ae1, loss_ae2, loss_dyn, loss_total = self.dynamics_losses(forward_pass, weight)

                    if weight[3] != 0:
                        x_final = x_final.to(self.device)
                        z_final = self.encoder(x_final)
                        loss_con = self.labels_losses(z_final, pairs, weight[3])
                        loss_contrastive_test += loss_con.item()
                
                    if weight[4] != 0:
                        loss_topo = self.topological_losses(forward_pass, weight)
                        loss_total += loss_topo
                        loss_topo_test += loss_topo.item()

                    loss_ae1_test += loss_ae1.item() 
                    loss_ae2_test += loss_ae2.item() 
                    loss_dyn_test += loss_dyn.item() 
                    epoch_test_loss += loss_total.item()

                epoch_test_loss /= num_batches

                self.test_losses['loss_ae1'].append(loss_ae1_test / num_batches)
                self.test_losses['loss_ae2'].append(loss_ae2_test / num_batches)
                self.test_losses['loss_dyn'].append(loss_dyn_test / num_batches)
                self.test_losses['loss_contrastive'].append(loss_contrastive_test / num_batches)
                self.test_losses['loss_topo'].append(loss_topo_test / num_batches)
                self.test_losses['loss_total'].append(epoch_test_loss)

            scheduler.step(epoch_test_loss)
            
            if epoch >= patience:
                if np.mean(self.test_losses['loss_total'][-patience:]) > np.mean(self.test_losses['loss_total'][-patience-1:-1]):
                    if self.verbose:
                        print("Early stopping")
                    break
            
            if self.verbose:
                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, epochs, epoch_train_loss, epoch_test_loss))

    def fine_tune(self, epochs=1000, patience=50, weight=[0,1,1,0,0]):
        '''
        Function that fine-tunes the encoder and decoder with the dynamics loss only.
        It will stop if the test loss does not improve for "patience" epochs.
        '''
        self.load_models()

        for param in self.encoder.parameters():
            param.requires_grad = False
        
        for param in self.decoder.parameters():
            param.requires_grad = False

        self.train(epochs, patience, weight)
        



