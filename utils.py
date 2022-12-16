import numpy as np
import faiss
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from KNN import KnnFGSM, KnnPGD
import torch.nn as nn
import torch.nn.functional as F

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)

def test_AUC(model, epoch, train_loader, test_loader, attack, device, attack_type):
  
    train_features = []
    k = 2
    test_distances = []
    test_distances_adv = []
    test_labels = []
    model.eval()

    print('Test Started ...')

    ###########################################################################

    print('Extracting Train Features...')


    with torch.no_grad():
      with tqdm(train_loader, unit="batch") as tepoch:
        for inputs, labels in tepoch:
            inputs = inputs.to(device)
            train_features += model.get_feature_vector(inputs).detach().cpu().numpy().tolist()

    print('Extracting Train Features Finished...')

    # features_tensor = torch.Tensor(train_features).to(device)
    train_features = np.array(train_features).astype(np.float32)
    ###########################################################################

    mean_train = torch.mean(torch.Tensor(train_features), axis=0)

    print('Extracting Test Features...')

    test_features = []
    adv_test_features = []
    adv_auc = 0
    test_labels_normal = []


    test_attack = None
    if attack_type == 'PGD':
        test_attack = KnnPGD.PGD_KNN(model, mean_train.to(device), eps=attack.eps, steps=attack.steps)
    else:
        test_attack = KnnFGSM.FGSM_KNN(model, mean_train.to(device), eps=attack.eps)

    
    with tqdm(test_loader, unit="batch") as tepoch:
      for inputs, labels in tepoch:
          inputs = inputs.to(device)
          labels = labels.to(device)
          test_features += model.get_feature_vector(inputs).detach().cpu().numpy().tolist()
          test_labels_normal += labels.detach().cpu().numpy().tolist()
          adv_inputs, labels, _, __ = test_attack(inputs, labels)
          adv_test_features += model.get_feature_vector(adv_inputs).detach().cpu().numpy().tolist()
          test_labels += labels.detach().cpu().numpy().tolist()

    test_features = np.array(test_features).astype(np.float32)
    adv_test_features = np.array(adv_test_features).astype(np.float32)


    test_distances = knn_score(train_features, test_features)
    test_distances_adv = knn_score(train_features, adv_test_features)



    print('Extracting Test Features Finished...')

    auc = roc_auc_score(test_labels_normal, test_distances)
    adv_auc = roc_auc_score(test_labels, test_distances_adv)

    print(f'AUC score on epoch {epoch} is: {auc * 100}')
    print(f'AUC Adversairal score on epoch {epoch} is: {adv_auc * 100}')

    return auc, adv_auc

def auc_softmax(model, epoch, test_loader, device):
  soft = torch.nn.Softmax(dim=1)
  anomaly_scores = []
  test_labels = []
  print('AUC Softmax Started ...')
  with torch.no_grad():
    with tqdm(test_loader, unit="batch") as tepoch:
      torch.cuda.empty_cache()
      for i, (data, target) in enumerate(tepoch):
        data, target = data.to(device), target.to(device)
        output = model(data)
        probs = soft(output).squeeze()
        anomaly_scores += probs[:, 1].detach().cpu().numpy().tolist()
        test_labels += target.detach().cpu().numpy().tolist()

  auc = roc_auc_score(test_labels, anomaly_scores)
  print(f'AUC - Softmax - score on epoch {epoch} is: {auc * 100}')
  return auc



def auc_softmax_adversarial(model, epoch, test_loader, attack, device):
  soft = torch.nn.Softmax(dim=1)
  anomaly_scores = []
  test_labels = []
  print('AUC Adversarial Softmax Started ...')
  with tqdm(test_loader, unit="batch") as tepoch:
    torch.cuda.empty_cache()
    for i, (data, target) in enumerate(tepoch):
      data, target = data.to(device), target.to(device)
      labels = target.to(device)
      adv_data = attack(data, target)
      output = model(adv_data)
      probs = soft(output).squeeze()
      anomaly_scores += probs[:, 1].detach().cpu().numpy().tolist()
      test_labels += labels.detach().cpu().numpy().tolist()

  auc = roc_auc_score(test_labels, anomaly_scores)
  print(f'AUC Adversairal - Softmax - score on epoch {epoch} is: {auc * 100}')
  return auc

def earlystop(model, data, target, perturb_steps, tau, epsilon=8/255, step_size=2/255, randominit_type= "uniform_randominit", loss_fn="cent", rand_init=True, omega=0):
    '''
    The implematation of early-stopped PGD
    Following the Alg.1 in our FAT paper <https://arxiv.org/abs/2002.11242>
    :param step_size: the PGD step size
    :param epsilon: the perturbation bound
    :param perturb_steps: the maximum PGD step
    :param tau: the step controlling how early we should stop interations when wrong adv data is found
    :param randominit_type: To decide the type of random inirialization (random start for searching adv data)
    :param rand_init: To decide whether to initialize adversarial sample with random noise (random start for searching adv data)
    :param omega: random sample parameter for adv data generation (this is for escaping the local minimum.)
    :return: output_adv (friendly adversarial data) output_target (targets), output_natural (the corresponding natrual data), count (average backword propagations count)
    '''
    model.eval()

    K = perturb_steps
    count = 0
    output_target = []
    output_adv = []
    output_natural = []

    control = (torch.ones(len(target)) * tau).cuda()

    # Initialize the adversarial data with random noise
    if rand_init:
        if randominit_type == "normal_distribution_randominit":
            iter_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
        if randominit_type == "uniform_randominit":
            iter_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda()
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
    else:
        iter_adv = data.cuda().detach()

    iter_clean_data = data.cuda().detach()
    iter_target = target.cuda().detach()
    output_iter_clean_data = model(data)

    while K > 0:
        iter_adv.requires_grad_()
        output = model(iter_adv)
        pred = output.max(1, keepdim=True)[1]
        output_index = []
        iter_index = []

        # Calculate the indexes of adversarial data those still needs to be iterated
        for idx in range(len(pred)):
            if pred[idx] != iter_target[idx]:
                if control[idx] == 0:
                    output_index.append(idx)
                else:
                    control[idx] -= 1
                    iter_index.append(idx)
            else:
                iter_index.append(idx)

        # Add adversarial data those do not need any more iteration into set output_adv
        if len(output_index) != 0:
            if len(output_target) == 0:
                # incorrect adv data should not keep iterated
                output_adv = iter_adv[output_index].reshape(-1, 3, 32, 32).cuda()
                output_natural = iter_clean_data[output_index].reshape(-1, 3, 32, 32).cuda()
                output_target = iter_target[output_index].reshape(-1).cuda()
            else:
                # incorrect adv data should not keep iterated
                output_adv = torch.cat((output_adv, iter_adv[output_index].reshape(-1, 3, 32, 32).cuda()), dim=0)
                output_natural = torch.cat((output_natural, iter_clean_data[output_index].reshape(-1, 3, 32, 32).cuda()), dim=0)
                output_target = torch.cat((output_target, iter_target[output_index].reshape(-1).cuda()), dim=0)

        # calculate gradient
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction='mean')(output, iter_target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(output_iter_clean_data, dim=1))
        loss_adv.backward(retain_graph=True)
        grad = iter_adv.grad

        # update iter adv
        if len(iter_index) != 0:
            control = control[iter_index]
            iter_adv = iter_adv[iter_index]
            iter_clean_data = iter_clean_data[iter_index]
            iter_target = iter_target[iter_index]
            output_iter_clean_data = output_iter_clean_data[iter_index]
            grad = grad[iter_index]
            eta = step_size * grad.sign()

            iter_adv = iter_adv.detach() + eta + omega * torch.randn(iter_adv.shape).detach().cuda()
            iter_adv = torch.min(torch.max(iter_adv, iter_clean_data - epsilon), iter_clean_data + epsilon)
            iter_adv = torch.clamp(iter_adv, 0, 1)
            count += len(iter_target)
        else:
            output_adv = output_adv.detach()
            return output_adv, output_target, output_natural, count
        K = K-1

    if len(output_target) == 0:
        output_target = iter_target.reshape(-1).squeeze().cuda()
        output_adv = iter_adv.reshape(-1, 3, 32, 32).cuda()
        output_natural = iter_clean_data.reshape(-1, 3, 32, 32).cuda()
    else:
        output_adv = torch.cat((output_adv, iter_adv.reshape(-1, 3, 32, 32)), dim=0).cuda()
        output_target = torch.cat((output_target, iter_target.reshape(-1)), dim=0).squeeze().cuda()
        output_natural = torch.cat((output_natural, iter_clean_data.reshape(-1, 3, 32, 32).cuda()),dim=0).cuda()
    output_adv = output_adv.detach()
    return output_adv, output_target


def get_data(model, exposure_iter, G, data, target, attack, device, config):
    model.eval()

    fake_data = None

    if config['use_gan']:
        y = G.sample_class(batch_size=data.shape[0])
        z = G.sample_latent(batch_size=data.shape[0])  
        x = G(z=z.to(device), y=y.to(device)).to(device)
        x = transforms.Resize((32, 32))(x)
        fake_data = (x - torch.min(x))/(torch.max(x)- torch.min(x))
    else:
        fake_data, fake_target = next(exposure_iter).to(device)
    

    if config['bw']:
        fake_data = transforms.Grayscale(3)(fake_data)

    new_data = torch.cat((fake_data, data))
    new_target = torch.cat((fake_target, target))
    
    if not clean:
        adv_data = attack(new_data, new_target)
        final_data = torch.cat((new_data, adv_data))
        final_target = torch.cat((new_target, new_target))
    else:
        final_data = new_data
        final_target = new_target
    
    model.train()
    return final_data, final_target
