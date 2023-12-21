import argparse
import os
import numpy as np 
import torch
import torch.nn.parallel
import torch.utils.data
import cv2
from data_utils.testDataset import testDataset
from data_utils.Dataset_RGBD_depth import EgoPAT3DDataset as RGBDDataset
from model.baseline import *
from model.depth_decoder import *
from model.pose_decoder import *
from model.resnet_encoder import *
import logging
from pathlib import Path
from tqdm import tqdm
from configs.cfg_utils import load_cfg
from loss import rgb_generatepred
import matplotlib.pyplot as plt



def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Baseline')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint')
    parser.add_argument('--model_epoch', default='', help='model name and epoch number')
    parser.add_argument('--config_file', default='', type=str, help='path to yaml config file')
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',default="mono_640x192")
    return parser.parse_args()


def main(cfg, model_epoch, checkpoint_path):
    '''HYPER PARAMETER'''
    model_name = model_epoch.split('/')[0]
    epoch_number = model_epoch.split('/')[1] # because model names are usually model_name/epoch_number


    
    '''CREATE DIR'''
    result_folder_path = f'./experiment/{model_name}/result/{epoch_number}' 
    experiment_dir = Path(f'./experiment/{model_name}/eval/{epoch_number}')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = experiment_dir
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger(cfg.MODEL.MODEL_NAME)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'eval_%s_cls.txt'%model_epoch.replace('/', '_'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------EVAL---------------------------------------------------')
    logger.info('PARAMETER ...')


    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = cfg.DATA.DATA_ROOT
    
    if cfg.MODEL.ARCH.POINTCLOUD == True and cfg.MODEL.ARCH.RGB == False:
        # Pointcloud
        TEST_DATASET = testDataset(cfg=cfg)
    else:
        # RGB
        TEST_DATASET = RGBDDataset(cfg, mode="annotest")
    
    finaltestDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=cfg.TESTING.BATCH_SIZE,shuffle=False)
    logger.info("The number of test data is: %d", len(TEST_DATASET))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    '''MODEL LOADING'''
    if cfg.MODEL.ARCH.POINTCLOUD == True and cfg.MODEL.ARCH.RGB == False:
        # Pointcloud
        classifier = Baseline(cfg=cfg).train()
    else:
        # Depth
        models = {}
        models["encoder"] = ResnetEncoder(18, True).train()

        models["depth"] = DepthDecoder(
        models["encoder"].num_ch_enc, [0, 1, 2, 3]).train()

    
    models["encoder"] = models["encoder"].to(device).eval()
    models["depth"] = models["depth"].to(device).eval()
  
    print('Load CheckPoint...')
    logger.info('Load CheckPoint')
    #need to cancel comment!!!!!!!!!!!!!!!!!!!!
    checkpoint_encoder = torch.load(checkpoint_path+'encoder.pth')
    checkpoint_depth = torch.load(checkpoint_path+'depth.pth')
    models["encoder"].load_state_dict(checkpoint_encoder['model_state_dict'])
    models["depth"].load_state_dict(checkpoint_depth['model_state_dict'])
    origin_writer = cv2.VideoWriter('./origin.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (224, 224))
    groundtruth_writer = cv2.VideoWriter('./ground_truth.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (224, 224))
    pred_writer = cv2.VideoWriter('./pred.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (224, 224))


    '''EVAL'''
    logger.info('Start evaluating...')
    print('Start evaluating...')
    
    # RGB related data
    err = 0
    terr = 0
    rgberr = 0
    total = 0
    ttotal = 0
    dist = []
    loss = 0
    count = 0

    with torch.no_grad():
        
        for batch_id, data in tqdm(enumerate(finaltestDataLoader, 0), total=len(finaltestDataLoader), smoothing=0.9):
            if cfg.MODEL.ARCH.POINTCLOUD == True and cfg.MODEL.ARCH.RGB == False:
                gt_xyz, pointcloud, motion , LENGTH,clipsource= data
                pointcloud=pointcloud.transpose(3,2)
                gt_xyz, pointcloud, motion = gt_xyz.to(device), pointcloud.to(device), motion.to(device)
                
                
                tic=cv2.getTickCount()

                pred = classifier(pointcloud[:,:,:3,:],
                                pointcloud[:,:,3:,:],
                                motion,
                                LENGTH.max().repeat(torch.cuda.device_count()).to(device),
                                cfg
                                )


                toc=cv2.getTickCount()-tic    
                toc /= cv2.getTickFrequency()
                print('speed:',LENGTH/toc,'FPS')
                scene_path = os.path.join(result_folder_path, clipsource[0][0],clipsource[1][0]) # path to save the results for each scene
                if not os.path.isdir(scene_path):
                    os.makedirs(scene_path)
                result_path=os.path.join(scene_path,clipsource[2][0]+'-'+clipsource[3][0]+'.txt') # path to save the predicted results for each clip
                gt_path=os.path.join(scene_path,clipsource[2][0]+'-'+clipsource[3][0]+'_gt.txt') # path to save the ground truth for each clip
                np.savetxt(gt_path,gt_xyz[0][:len(pred)].cpu().numpy())

                with open(result_path, 'w') as f:
                    for xx in pred:
                        
                        def dcon(x):
                            resultlist=torch.linspace(-1,1,1024*5).cuda()
                            x=x/x.max()

                            x[torch.where(x<=0.5)]=0

                            return (x*resultlist).sum()/x.sum()
    

                        data=str(float(dcon(xx[0][0])))+','+str(float(dcon(xx[0][1])))+','+str(float(dcon(xx[0][2])))
                        f.write(data+'\n')
            
            else:
                gt_xyz, rgb, depth, rangenum, finalsource, hand, handLM = data

                rgb = rgb.transpose(3,4)
                rgb = rgb.transpose(2,3)
                rgb, depth, gt_xyz, handLM = rgb.to(device), depth.to(device), gt_xyz.to(device), handLM.to(device)
                
                tic=cv2.getTickCount()

                for batch_id in range(depth.shape[0]):
                    pred = models["depth"](models["encoder"](rgb[batch_id,:,:,:]))

                    pred_disp, pred_depth = disp_to_depth(pred[("disp", 0)], 1e1, 1e3)
                    for id in range(rangenum[batch_id]):
                        deep = depth[batch_id,id,:,:]
                        # pred_disp = pred_disp.cpu()[:, 0].numpy()
                        mask = deep.clone()
                        mask[torch.where(mask<10)] = 0
                        mask[torch.where(mask>1000)] = 0
                        mask[torch.where(mask>=10)] = 1
                        masks = mask.sum()
                        if masks == 0:
                            continue
                        # print(pred_depth[id])
                        loss += (torch.abs(pred_depth[id]-deep)*mask).sum()/masks
                        count += 1
                        frame = rgb[batch_id,id].cpu()
                        frame = frame.transpose(0,1)
                        frame = frame.transpose(1,2)
                        frame = frame.numpy().astype(np.uint8)
                        deep = (deep.cpu().squeeze().numpy()/4).astype(np.uint8)
                        pred_frame = (pred_depth[id].cpu().squeeze().numpy()/4).astype(np.uint8)
                        frame=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        deep=cv2.cvtColor(deep, cv2.COLOR_GRAY2RGB)
                        # print(pred_frame.shape)
                        pred_frame=cv2.cvtColor(pred_frame, cv2.COLOR_GRAY2RGB)
                        origin_writer.write(frame)
                        groundtruth_writer.write(deep)
                        pred_writer.write(pred_frame)



                toc=cv2.getTickCount()-tic    
                toc /= cv2.getTickFrequency()
                print('speed:',rangenum/toc,'FPS')
                # scene_path = os.path.join(result_folder_path, finalsource[0][0],finalsource[1][0]) # path to save the results for each scene
                # if not os.path.isdir(scene_path):
                #     os.makedirs(scene_path)
                # result_path=os.path.join(scene_path,str(int(finalsource[2][0]))+'-'+str(int(finalsource[3][0]))+'.txt') # path to save the predicted results for each clip
                # gt_path=os.path.join(scene_path,str(int(finalsource[2][0]))+'-'+str(int(finalsource[3][0]))+'_gt.txt') # path to save the ground truth for each clip
                # np.savetxt(gt_path,gt_xyz[0][:len(pred)].cpu().numpy())
                
                # if cfg.TRAINING.LOSS == 'RGB_Ori':
                #     with open(result_path, 'w') as f:
                #         id = 0
                #         ma = 0
                #         prev = 0
                #         end = False
                #         for xx in pred:
                #             xx = xx.detach()
                #             pos,handx,handy,tim = rgb_generatepred(xx[0])
                #             # print(hand[0][id])
                #             # u=pos[0]
                #             # v=pos[1]
                #             u=(pos[0]*1.80820276e+03/pos[2]+1.94228662e+03)/3840
                #             v=(pos[1]*1.80794556e+03/pos[2]+1.12382178e+03)/2160
                #             if hand[0][id][0]!=0 and hand[0][id][1]!=0:
                #                 err += np.sqrt(float((handx-hand[0][id][0])**2+(handy-hand[0][id][1])**2))
                #                 total += 1
                #             if id > 0 and hand[0][id-1][0]!=0 and hand[0][id-1][1]!=0 and hand[0][id][0]!=0 and hand[0][id][1]!=0:
                #                 diff = hand[0][id]-prev
                #                 diff = diff[0]**2+diff[1]**2
                #                 if diff >ma:
                #                     ma = diff
                #                 print(diff/ma)
                #                 # ma = diff
                #                 if not end and diff/ma<0.05:
                #                     end = True
                #                     dist.append(int(rangenum[0]-id))
                #                 if id >=10:
                #                     u = u*diff/ma+hand[0][id][0]*(1-diff/ma)
                #                     v = v*diff/ma+hand[0][id][1]*(1-diff/ma)
                #                     # if depth[0,int(max(min(v*2160,2159),0)),int(max(min(u*3840,3839),0))]!=0:
                #                     #     z1 = depth[0,int(max(min(v*2160,2159),0)),int(max(min(u*3840,3839),0))]/1000
                #                     # else:
                #                     z1 = pos[2]
                #                     # z1 = get_depth(depth[0],int(hand[0][id][1]*3840),int(hand[0][id][0]*3840))/1000
                #                     x1 = (int(hand[0][id][0]*3840)-1.94228662e+03)*z/1.80820276e+03
                #                     y1 = (int(hand[0][id][1]*2160)-1.12382178e+03)*z/1.80794556e+03
                #                     pos[0] = pos[0]*diff/ma+x1*(1-diff/ma)
                #                     pos[1] = pos[1]*diff/ma+y1*(1-diff/ma)
                #                     pos[2] = pos[2]*diff/ma+z1*(1-diff/ma)
                #             # terr += np.sqrt(float((tim-id/rangenum[0])**2))
                #             # rgberr += np.sqrt(float((u-gt_xy[0][id][0])**2+(v-gt_xy[0][id][1])**2))
                #             # ttotal += 1
                #             u = int(u*3840)
                #             v = int(v*2160)
                #             if u>=3840:
                #                 u=3840
                #             if u<0:
                #                 u=0
                #             if v>=2160:
                #                 u=2160
                #             if v<0:
                #                 v=0
                #             # z = get_depth(depth[0],v,u)/1000
                #             # if depth[0,int(max(min(v,2159),0)),int(max(min(u,3839),0))]!=0:
                #             #     z = depth[0,int(max(min(v,2159),0)),int(max(min(u,3839),0))]/1000
                #             # else:
                #             z = pos[2]
                #             x = (u-1.94228662e+03)*z/1.80820276e+03
                #             y = (v-1.12382178e+03)*z/1.80794556e+03

                #             data=str(float(x))+','+str(float(y))+','+str(float(z))
                #             # data=str(float(pos[0]))+','+str(float(pos[1]))+','+str(float(pos[2]))
                #             f.write(data+'\n')
                #             if hand[0][id][0]!=0 and hand[0][id][1]!=0:
                #                 prev = hand[0][id]
                #             id+=1 

                # elif cfg.TRAINING.LOSS == 'Ori':
                #     with open(result_path, 'w') as f:
                #         for xx in pred:
                            
                #             def dcon(x):
                #                 resultlist=torch.linspace(-1,1,1024*5).cuda()
                #                 x=x/x.max()

                #                 x[torch.where(x<=0.5)]=0

                #                 return (x*resultlist).sum()/x.sum()
    
                #             data=str(float(dcon(xx[0][0])))+','+str(float(dcon(xx[0][1])))+','+str(float(dcon(xx[0][2])))
                #             f.write(data+'\n')

#     state = {
#         'epoch': model_epoch,
# #        'train_accuracy': train_accuracy,
# #        'test_accuracy': test_accuracy,
#         'model_state_dict': classifier.state_dict()
#     }
#     savepath  = './experiment/' + model_name + '/result/' + epoch_number + '/checkpoint.pth'
    # torch.save(state, savepath)   
    # print("dist",len(dist),dist)
    # print(np.histogram(dist,bins=range(30),density=True))
    # dist = np.asarray(dist)
    # hist = np.histogram(dist,bins=range(31),density=True)
    # x = np.linspace(0,29, num=30)
    # plt.plot(x,hist[0])
    # plt.savefig("./dist.jpg")
                
    # if cfg.MODEL.ARCH.POINTCLOUD == False and cfg.MODEL.ARCH.RGB == True:
    #     logger.info(err/total)
    #     logger.info(terr/ttotal)
    #     logger.info(rgberr/ttotal)   
    origin_writer.release()
    groundtruth_writer.release()
    pred_writer.release()
    print("loss",loss/count)
    logger.info('End of evaluation...')

if __name__ == '__main__':
    arg = parse_args()
    cfg = load_cfg(arg.config_file)
    model_epoch = arg.model_epoch
    checkpoint_path = arg.checkpoint
    main(cfg, model_epoch, checkpoint_path)
