import torch as t
from utils.layers import *
from torchvision import transforms

def generatepred(x):
        resultlist=t.linspace(-1,1,1024*5).cuda()
        x=x/x.max(1)[0].unsqueeze(-1)
        for i in range(3):
            x[i][t.where(x[i]<0.5)]=0
        return (x*resultlist).sum(1)/x.sum(1)

def rgb_generatepred_manual(x):
    '''
    rgb loss that's manual during inference
    '''
    resultlist=t.linspace(-1,1,1024*5).cuda()
    x=x/x.max(1)[0].unsqueeze(-1)
    for i in range(4):
            x[i][t.where(x[i]<0.5)]=0
    return (x[:3]*resultlist).sum(1)/x[:3].sum(1),(x[3]*resultlist).sum(0)/x[3].sum(0),(x[4]*resultlist).sum(0)/x[4].sum(0),(x[5]*resultlist).sum(0)/x[5].sum(0)

def rgb_generatepred(x):
    '''
    rgb loss that's incorporated into the training
    '''
    resultlist=t.linspace(-1,1,1024*5).cuda()
    x=x/x.max(1)[0].unsqueeze(-1)
    for i in range(6):
            x[i][t.where(x[i]<0.5)]=0
    return (x[:3]*resultlist).sum(1)/x[:3].sum(1),(x[3]*resultlist).sum(0)/x[3].sum(0),(x[4]*resultlist).sum(0)/x[4].sum(0),(x[5]*resultlist).sum(0)/x[5].sum(0)


def calculate(x,y):

        pred=generatepred(x)
        loss=((pred-y)**2).sum()

        return loss

def rgb_calculate_manual(x,y,hand,time,train):

        pred,handx,handy,pred_time=rgb_generatepred_manual(x)
        loss=((pred[0]-y[0])**2+(pred[1]-y[1])**2+(pred[2]-y[2])**2).sum()
        if train:
            loss+=0.1*((pred_time-time)**2).sum()
            if hand[0] != 0 and hand[1] != 0:
                loss += 0.1*((handx-hand[0])**2+(handy-hand[1])**2).sum()

        return loss


def rgb_calculate(x,y,hand,time,train,pre,id):

        pred,handx,handy,pred_time=rgb_generatepred(x)
        rate = 1
        if id >= 10:
            if hand[id][0] != 0 and hand[id][1] != 0 and hand[id-1][0] != 0 and hand[id-1][1] != 0:
                ma = 0
                for i in range(id-1):
                    diff = hand[i+1]-hand[i]
                    diff = diff[0]**2+diff[1]**2
                    if diff>ma:
                        ma = diff
                rate = diff/ma

        loss=((pred[0]-y[0])**2+(pred[1]-y[1])**2+(pred[2]-y[2])**2).sum()
        if train:
            loss+=0.1*((pred_time-time)**2).sum()
            # if hand[id][0] != 0 and hand[id][1] != 0:
            #     loss += 0.1*((handx-hand[id][0])**2+(handy-hand[id][1])**2).sum()

        return loss,pred,pred_time,handx,handy

def oriloss(pred,gt,length,device):
    batch=gt.size()[0]
    loss=[]
    for i in range(batch):
        
        for pred_xyz in range(length[i]):
            loss.append((calculate(pred[pred_xyz][i],gt[i][pred_xyz])*(2-pred_xyz/length[i])))
    return sum(loss)/batch

def last_oriloss(pred,gt,length,device):
    batch=gt.size()[0]
    loss=[]
    for i in range(batch):
        
        for pred_xyz in range(length[i]):
            if pred_xyz==length[i]-1: # only calculate the loss for the last frame
                loss.append((calculate(pred[pred_xyz][i],gt[i][pred_xyz])*(2-pred_xyz/length[i])))
    return sum(loss)/batch

def scaled_oriloss(pred,gt,length,device):
    batch=gt.size()[0]
    loss=[]
    max_length = max(length) # the maximum sequence length in the batch
    for i in range(batch):
        
        for pred_xyz in range(length[i]):
            ori_loss = (calculate(pred[pred_xyz][i],gt[i][pred_xyz])*(2-pred_xyz/length[i]))
            scaled_loss = ori_loss*(max_length/length[i]) # scale the loss by the maximum sequence length
            loss.append(scaled_loss)
    return sum(loss)/batch

def last_frame_loss(pred,gt):
    batch=gt.size()[0]
    loss=[]
    for i in range(batch):

        loss.append(calculate(pred[0][i],gt[i,:]))
    return 1000*sum(loss)/batch

def last_frame_dist(pred,gt):
    batch=gt.size()[0]
    loss=[]
    for i in range(batch):

        loss.append(t.sqrt(calculate(pred[0][i],gt[i,:])))
    return sum(loss)/batch

def own_l2_loss(pred,gt,length):
    '''
    pred: (seq_len, batch, 1024)
    gt: (seq_len, batch, 1024)
    length: (batch)
    '''
    batch=gt[0].shape[0] # batch size
    loss=[]
    for i in range(batch):
        
        # the rest after length[i] are 0-padded
        sequence_loss = []
        for j in range(length[i]):
            adjusted_loss = t.nn.functional.mse_loss(pred[j][i], gt[j][i], reduction='mean')*(2-j/length[i]) # adjust the loss by the sequence length
            sequence_loss.append(adjusted_loss)
        sequence_loss = t.as_tensor(sequence_loss) # cast to tensor to preserve the autograd graph
        sequence_loss = t.sum(sequence_loss)/length[i] # average the loss over the sequence
        loss.append(sequence_loss)
    loss = t.as_tensor(loss) # cast to tensor to preserve the autograd graph
    return t.sum(loss)/batch

def rgbloss_manual(pred,gt,hand,length,device,train=True):
    batch=gt.size()[0]
    loss=[]
    for i in range(batch):
        
        for pred_xyz in range(length[i]):
            loss.append(25*(rgb_calculate_manual(pred[pred_xyz][i],gt[i][pred_xyz],hand[i][pred_xyz],pred_xyz/length[i],train)*(2-pred_xyz/length[i]))/length[i])
            if loss[-1]>100:
                print("large",length[i],loss[-1],gt[i][pred_xyz])
    return sum(loss)/(batch)

def rgbloss(pred,gt,hand,length,device,train=True):
    batch=gt.size()[0]
    loss=[]
    pre = []
    for i in range(batch):
        
        for pred_xyz in range(length[i]):
            single,pres,pred_time,handx,handy = rgb_calculate(pred[pred_xyz][i],gt[i][pred_xyz],hand[i],pred_xyz/length[i],train,pre,pred_xyz)
            loss.append(25*(single*(2-pred_xyz/length[i]))/length[i])
            if loss[-1]>100:
                print("large",length[i],loss[-1],gt[i][pred_xyz],hand[i][pred_xyz],handx,handy,pred_time,pred_xyz)
            pre.append(pres)
    return sum(loss)/(batch)

def compute_reprojection_loss(pred, target):
    """Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(0, True)

    ssim = SSIM().cuda()

    ssim_loss = ssim(pred, target).mean(0, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss

def mono_loss(rgb, outputs, rangenum, device):
    """Compute the reprojection and smoothness losses for a minibatch
    """
    losses = {}
    total_loss = 0
    inputs = rgb.clone()

    for batch_id in range(rgb.shape[0]):
        for scale in range(4):
            loss = 0
            reprojection_losses = []

            source_scale = 0
            inputs = rgb[batch_id,:,:,:].clone()

            disp = outputs[("disp", scale)][batch_id]
            target = rgb[batch_id,0,:,:].clone()
            rate = 2**scale
            p = transforms.Compose([transforms.Resize((224//rate,224//rate))])
            size = inputs.shape[0]
            inputs = inputs.contiguous().view(inputs.shape[0]*3,224,224)
            inputs = p(inputs)
            inputs = inputs.contiguous().view(size,3,224//rate,224//rate)
            color = inputs[0,:,:]

            for id in range(rangenum[batch_id]-1):
                frame_id = id+1
                pred = outputs[("color", frame_id, scale)][batch_id]
                p1 = transforms.Compose([transforms.Resize((224,224))])
                pred = p1(pred)
                reprojection_losses.append(compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 0)

            identity_reprojection_losses = []
            for id in range(rangenum[batch_id]-1):
                frame_id = id+1
                pred = inputs[frame_id,:,:]
                p1 = transforms.Compose([transforms.Resize((224,224))])
                pred = p1(pred)
                identity_reprojection_losses.append(
                    compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 0)

            identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses
            # print(identity_reprojection_loss.shape, reprojection_loss.shape, outputs[("color", frame_id, scale)].shape)

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=device) * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=0)

            if combined.shape[0] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=0)

            outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[0] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(0, True).mean(1, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += 0.1*smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

    total_loss /= 4
    losses["loss"] = total_loss
    return losses