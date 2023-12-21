import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import logging
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

from data_utils.trainDataset import trainDataset
from data_utils.validateDataset import validateDataset
from data_utils.Dataset_RGBD_depth import EgoPAT3DDataset as RGBDDataset
from model.baseline import *
from model.depth_decoder import *
from model.pose_decoder import *
from model.resnet_encoder import *
# from model.baseline_streaming import *
from loss import oriloss, last_oriloss, rgbloss, rgbloss_manual, mono_loss
from utils.utils import save_checkpoint
from utils.layers import *
from configs.cfg_utils import load_cfg
from torchvision import transforms



def blockprint():
    sys.stdout = open(os.devnull, 'w')    

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Baseline')
    
    parser.add_argument(
    '--config_file',
    default='',
    type=str,
    help='path to yaml config file',)
    return parser.parse_args()

def get_my_loss(cfg, **kwargs):
    '''
    Wrapper for the loss function. 
    So now when I modify loss function I only need to modify this function instead of in both train and val function
    '''
    if cfg.TRAINING.LOSS == 'Ori':
        criterion = kwargs['criterion']
        loss = criterion(pred=kwargs['pred'],
                        gt=kwargs['gt_xyz'],
                        length=kwargs['LENGTH'],
                        device=kwargs['device'])

    elif cfg.TRAINING.LOSS == 'RGB_Ori':
        # criterion = kwargs['criterion']
        # loss = criterion(pred=kwargs['pred'],
        #                 gt=kwargs['gt_xyz'],
        #                 hand=kwargs['hand'],
        #                 length=kwargs['LENGTH'],
        #                 device=kwargs['device'],
        #                 train=kwargs['train'])
        loss = mono_loss(rgb=kwargs['rgb'],outputs=kwargs['pred'],rangenum=kwargs['rangenum'],device=kwargs['device'])
    else:
        raise NotImplementedError('Not implemented loss')
    return loss

def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M

def predict_poses(rgb, features, pose, encoder):
    """Predict poses between input frames for monocular sequences.
    """
    outputs = {}
    pose_feats = {f_i: rgb[:,f_i,:,:] for f_i in range(rgb.shape[1])}

    for f_i in range(rgb.shape[1]):
        if f_i != 0:
            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_inputs = [encoder(torch.cat(pose_inputs, 1))]

            axisangle, translation = pose(pose_inputs)
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

    return outputs

def generate_images_pred(rgb, outputs):
    """Generate the warped (reprojected) color images for a minibatch.
    Generated images are saved into the `outputs` dictionary.
    """
    for scale in range(4):
        inputs = rgb.clone()
        rate = 2**scale
        p = transforms.Compose([transforms.Resize((224//rate,224//rate))])
        size = inputs.shape[1]
        inputs = inputs.contiguous().view(inputs.shape[0],inputs.shape[1]*3,224,224)
        inputs = p(inputs)
        inputs = inputs.contiguous().view(inputs.shape[0],size,3,224//rate,224//rate)
        disp = outputs[("disp", scale)]
        p1 = transforms.Compose([transforms.Resize((224,224))])
        disp = p1(disp)
        source_scale = 0

        _, depth = disp_to_depth(disp, 1e-2, 1)
        # print(depth[0])
        # print(depth.shape)

        outputs[("depth", 0, scale)] = depth

        for id in range(rgb.shape[1]-1):
            frame_id = id+1
            K = np.zeros((rgb.shape[0],4,4))
            inv_K = np.zeros((rgb.shape[0],4,4))
            T = outputs[("cam_T_cam", 0, frame_id)]

            cx = 1.94228662e+03 / 3840
            cy = 1.12382178e+03 / 2160
            fx = 1.80820276e+03 / 3840
            fy = 1.80794556e+03 / 2160

            temp_K =np.array([[fx, 0., cx, 0.],
                                [0., fy, cy, 0.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]], dtype="float32")
            # temp_K[0, :] *= 224 // (2 ** scale)
            # temp_K[1, :] *= 224 // (2 ** scale)
            temp_K[0, :] *= 224
            temp_K[1, :] *= 224
            temp_inv_K = np.linalg.pinv(temp_K)
            for id in range(rgb.shape[0]):
                K[id] = temp_K
                inv_K[id] = temp_inv_K
            K = torch.from_numpy(K).float().cuda()
            inv_K = torch.from_numpy(inv_K).float().cuda()

            # backproject_depth = BackprojectDepth(rgb.shape[0], 224 // (2 ** scale), 224 // (2 ** scale)).cuda()
            # project_3d = Project3D(rgb.shape[0], 224 // (2 ** scale), 224 // (2 ** scale)).cuda()
            backproject_depth = BackprojectDepth(rgb.shape[0], 224, 224).cuda()
            project_3d = Project3D(rgb.shape[0], 224, 224).cuda()

            cam_points = backproject_depth(
                depth, inv_K)
            pix_coords = project_3d(
                cam_points, K, T)

            outputs[("sample", frame_id, scale)] = pix_coords

            outputs[("color", frame_id, scale)] = F.grid_sample(
                inputs[:,frame_id,:,:,:].float(),
                outputs[("sample", frame_id, scale)],
                padding_mode="border",
                align_corners=True)

            outputs[("color_identity", frame_id, scale)] = \
                inputs[:,frame_id,:,:,:].float()

def get_my_pred(cfg, **kwargs):
    '''
    Wrapper for the prediction function. 
    So now when I modify prediction function I only need to modify this function instead of in both train and val function
    '''
    if cfg.MODEL.STREAMING == True:
        # Streaming
        pred_list = []
        hout, cout = 0, 0
        classifier = kwargs['classifier']

        # the first frame
        pred, hout, cout = classifier(img = kwargs['rgb'][:,0,:,:],
                          hand = kwargs['handLM'][:,0,:],
                          start = True,
                          hout = hout,
                          cout = cout,
                          cfg = cfg
                          )
        pred_list.append(pred)

        # the rest frames
        for idx in range(1, int(kwargs['LENGTH'][0])): # the first frame has been processed
            pred, hout, cout = classifier(img = kwargs['rgb'][:,idx,:,:],
                          hand = kwargs['handLM'][:,idx,:],
                          start = False,
                          hout = hout,
                          cout = cout,
                          cfg = cfg
                          )
            pred_list.append(pred)


    else:
        if cfg.MODEL.ARCH.POINTCLOUD == True and cfg.MODEL.ARCH.RGB == False:
            classifier = kwargs['classifier']
            pred_list = classifier(pointxyz = kwargs['pointcloud'][:,:,:3,:], 
                            pointfeat = kwargs['pointcloud'][:,:,3:,:],
                            motion = kwargs['motion'],
                            LEGHTN = kwargs['LENGTH'].max().repeat(torch.cuda.device_count()).to(kwargs['device']),
                            cfg = cfg
                            )
        else:
            # classifier = kwargs['classifier']
            # pred_list = classifier(img = kwargs['rgb'],
            #                 hand = kwargs['handLM'],
            #                 LEGHTN=kwargs['LENGTH'],
            #                 cfg = cfg
            #                 )
            encoder=kwargs["encoder"]
            depth=kwargs["depth"]
            pose_encoder=kwargs["pose_encoder"]
            pose=kwargs["pose"]
            features = encoder(kwargs['rgb'][:,0,:,:])
            outputs = depth(features)

            outputs.update(predict_poses(kwargs['rgb'], features, pose, pose_encoder))

            generate_images_pred(kwargs['rgb'], outputs)
            pred_list = outputs

    return pred_list

def train(encoder, depth_decoder, pose_encoder, pose, dataloader, optimizer, criterion, scheduler, scaler, device, global_rank, cfg):
    encoder.train()
    depth_decoder.train()
    pose_encoder.train()
    pose.train()
    total_loss = 0
    scheduler.step()
    for batch_id, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), smoothing=0.9, disable=global_rank!=0):
        # disable the progress bar for all processes except the first one
        
        if cfg.MODEL.ARCH.POINTCLOUD == True and cfg.MODEL.ARCH.RGB == False:
            # pointcloud
            gt_xyz,pointcloud, motion, LENGTH, _ = data
            pointcloud=pointcloud.transpose(3,2)
            gt_xyz, pointcloud, motion = gt_xyz.to(device),pointcloud.to(device), motion.to(device)

            optimizer.zero_grad()
            pred = get_my_pred(cfg=cfg,
                                classifier=classifier, 
                                pointcloud=pointcloud, 
                                motion=motion, 
                                LENGTH=LENGTH, 
                                device=device,
                                )

            loss = get_my_loss(cfg=cfg, 
                                pred=pred, 
                                gt_xyz=gt_xyz,
                                LENGTH=LENGTH,
                                device=device, 
                                criterion=criterion)
        else:
            # rgb
            gt_xyz, rgb, depth, rangenum, finalsource, hand, handLM = data


            rgb = rgb.transpose(3,4)
            rgb = rgb.transpose(2,3)
            rgb, depth, gt_xyz, handLM = rgb.to(device), depth.to(device), gt_xyz.to(device), handLM.to(device)
            
            optimizer.zero_grad()

            pred = get_my_pred(cfg=cfg,
                                encoder=encoder,
                                depth=depth_decoder,
                                pose_encoder=pose_encoder,
                                pose=pose,
                                rgb=rgb,
                                handLM=handLM,
                                LENGTH=[25],
                                device=device,
                                )
            
            loss = get_my_loss(cfg=cfg,
                                pred=pred,
                                gt_xyz=gt_xyz,
                                rangenum=rangenum,
                                hand=hand,
                                LENGTH=rangenum,
                                device=device,
                                criterion=criterion,
                                train=True,
                                rgb=rgb,
                                )

        depth_loss = 0

        for batch_id in range(depth.shape[0]):
            pred = depth_decoder(encoder(rgb[batch_id,:,:,:]))

            pred_disp, pred_depth = disp_to_depth(pred[("disp", 0)], 1e-2, 1)
            pred_depth = 1000*pred_depth
            for id in range(rangenum[batch_id]):
                deep = depth[batch_id,id,:,:]
                # print(deep)
                # pred_disp = pred_disp.cpu()[:, 0].numpy()
                mask = deep.clone()
                mask[torch.where(mask<10)] = 0
                mask[torch.where(mask>1000)] = 0
                mask[torch.where(mask>10)] = 1
                masks = mask.sum()
                if masks < 10:
                    continue
                depth_loss += 1e-6*(((pred_depth[id]-deep)**2)*mask).sum()/masks

        loss["loss"] -= loss["loss"]
        loss["loss"] += depth_loss

        total_loss = loss["loss"] + total_loss

        loss["loss"].backward()
        optimizer.step()

    
    # it's not appropriate to average the loss, because the different stages should have different loss
    return total_loss


def validate(encoder, depth_decoder, pose_encoder, pose, dataloader, criterion, scaler, device, global_rank, cfg):
    encoder.eval()
    depth_decoder.eval()
    pose_encoder.eval()
    pose.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch_id, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), smoothing=0.9, disable=global_rank!=0):
            # disable the progress bar for all processes except the first one
            
            if cfg.MODEL.ARCH.POINTCLOUD == True and cfg.MODEL.ARCH.RGB == False:
                # pointcloud
                gt_xyz,pointcloud, motion, LENGTH, _ = data
                pointcloud=pointcloud.transpose(3,2)
                gt_xyz, pointcloud, motion = gt_xyz.to(device),pointcloud.to(device), motion.to(device)

                pred = get_my_pred(cfg=cfg,
                                    classifier=classifier, 
                                    pointcloud=pointcloud, 
                                    motion=motion, 
                                    LENGTH=LENGTH, 
                                    device=device,
                                    )

                loss = get_my_loss(cfg=cfg, 
                                    pred=pred, 
                                    gt_xyz=gt_xyz,
                                    LENGTH=LENGTH,
                                    device=device, 
                                    criterion=criterion,)
            else:
                # rgb
                gt_xyz, rgb, depth, rangenum, finalsource, hand, handLM = data

                rgb = rgb.transpose(3,4)
                rgb = rgb.transpose(2,3)
                rgb, depth, gt_xyz, handLM = rgb.to(device), depth.to(device), gt_xyz.to(device), handLM.to(device)
                
                # pred = get_my_pred(cfg=cfg,
                #                     encoder=encoder,
                #                     depth=depth_decoder,
                #                     pose_encoder=pose_encoder,
                #                     pose=pose,
                #                     rgb=rgb,
                #                     handLM=handLM,
                #                     LENGTH=[25],
                #                     device=device,
                #                     )
                loss = 0
                for batch_id in range(depth.shape[0]):
                    pred = depth_decoder(encoder(rgb[batch_id,:,:,:]))

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
                
                # loss = get_my_loss(cfg=cfg,
                #                     pred=pred,
                #                     gt_xyz=gt_xyz,
                #                     hand=hand,
                #                     LENGTH=rangenum,
                #                     device=device,
                #                     criterion=criterion,
                #                     train=False
                #                     )

            total_loss = loss + total_loss


    
    # it's not appropriate to average the loss, because the different stages should have different loss
    return total_loss/(count)


def main(cfg):
    '''DDP SETUP'''
    dist.init_process_group(backend="nccl") # init distributed
    slurm_proc_id = os.environ.get("SLURM_PROCID", None)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    print(f'process started with local rank: {local_rank}, global rank: {global_rank}, world size: {world_size}')

    '''CREATE DIR'''
    basepath=os.getcwd()
    experiment_dir = Path(os.path.join(basepath,'experiment'))
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s'%cfg.MODEL.MODEL_NAME)
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    # a place to save the evaluation results and sbatch stdouts
    eval_dir = file_dir.joinpath('eval/')
    eval_dir.mkdir(exist_ok=True)
    output_logs_dir = eval_dir.joinpath('output_logs/')
    output_logs_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger(cfg.MODEL.MODEL_NAME)
    if global_rank == 0: # only log on the first process
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(file_dir.joinpath('train_%s_cls.txt'%cfg.MODEL.MODEL_NAME))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info('---------------------------------------------------TRANING---------------------------------------------------')



    '''DATA LOADING'''
    if cfg.MODEL.ARCH.POINTCLOUD == True and cfg.MODEL.ARCH.RGB == False:
        # Pointcloud
        TRAIN_DATASET = trainDataset(cfg) 
        VAL_DATASET = validateDataset(cfg) 
    else:
        # RGB
        TRAIN_DATASET = RGBDDataset(cfg, mode="annotrain")
        VAL_DATASET = RGBDDataset(cfg, mode="annovalidate")

    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(TRAIN_DATASET, shuffle=True)
        val_sampler = DistributedSampler(VAL_DATASET, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_iterator = DataLoader(TRAIN_DATASET, 
                                batch_size=cfg.DATA.DATA_LOADER.BATCH_SIZE,
                                num_workers=cfg.DATA.DATA_LOADER.NUM_WORKERS,
                                sampler=train_sampler,
                                drop_last=True,
                                pin_memory=cfg.DATA.DATA_LOADER.PIN_MEMORY)
    val_iterator = DataLoader(VAL_DATASET, 
                                batch_size=int(cfg.DATA.DATA_LOADER.BATCH_SIZE*2), 
                                num_workers=cfg.DATA.DATA_LOADER.NUM_WORKERS,
                                sampler=val_sampler,
                                drop_last=True,
                                pin_memory=cfg.DATA.DATA_LOADER.PIN_MEMORY)
    
    if global_rank == 0: # only log on the first process
        logger.info("The number of training data is: %d", len(TRAIN_DATASET))


    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    '''MODEL LOADING'''
    if cfg.MODEL.STREAMING == True:
        # Streaming
        classifier = Baseline_RGB_Streaming(cfg=cfg).train()

    else:
        # Non-streaming
        if cfg.MODEL.ARCH.POINTCLOUD == True and cfg.MODEL.ARCH.RGB == False:
            # Pointcloud
            classifier = Baseline(cfg=cfg).train()
        else:
            # RGB
            classifier = Baseline_RGB(cfg=cfg).train()
            models = {}
            parameters_to_train = []

            models["encoder"] = ResnetEncoder(18, True)
            parameters_to_train += list(models["encoder"].parameters())

            models["depth"] = DepthDecoder(
            models["encoder"].num_ch_enc, [0, 1, 2, 3])
            parameters_to_train += list(models["depth"].parameters())

            models["pose_encoder"] = ResnetEncoder(
            18,
            True,
            num_input_images=2)
            parameters_to_train += list(models["pose_encoder"].parameters())

            models["pose"] = PoseDecoder(
                models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)
            parameters_to_train += list(models["pose"].parameters())


    if dist.is_available() and dist.is_initialized():
        device = f"cuda:{local_rank}"
        classifier = classifier.to(device)
        classifier = DDP(classifier, device_ids=None)
        for model in models.values():
            model.to(device)
            model = DDP(model, device_ids=None)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier = classifier.to(device)
        for model in models.values():
            model.to(device)

    for model in models.values():
        torch.compile(model)

    if cfg.MODEL.CHECKPOINT != '':
        if global_rank == 0: # only log on the first process
            print('Use pretrain model...')
            logger.info('Use pretrain model')
        start_epoch = torch.load(cfg.MODEL.CHECKPOINT)['epoch']
        classifier.module.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT)['model_state_dict'])
    else:
        if global_rank == 0: # only log on the first process
            print('No existing model, starting training from scratch...')
        start_epoch = 0

    '''OPTIMIZER, LOSS, SCHEDULER'''
    # Optimizer
    if cfg.TRAINING.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(classifier.parameters(), lr=cfg.TRAINING.LEARNING_RATE, momentum=0.9, weight_decay=cfg.TRAINING.DECAY_RATE)
    elif cfg.TRAINING.OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(parameters_to_train, lr=cfg.TRAINING.LEARNING_RATE, betas=(0.9, 0.999), weight_decay=cfg.TRAINING.DECAY_RATE)
    else:
        raise NotImplementedError('Not implemented optimizer')
    
    # Loss
    if cfg.TRAINING.LOSS == 'Ori':
        criterion = oriloss
    elif cfg.TRAINING.LOSS == 'Last_Ori':
        criterion = last_oriloss
    elif cfg.TRAINING.LOSS == 'RGB_Ori_Manual':
        criterion = rgbloss_manual
    elif cfg.TRAINING.LOSS == 'RGB_Ori':
        criterion = rgbloss

    else:
        raise NotImplementedError('Not implemented loss')
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)


    '''TRANING'''
    if global_rank == 0: # only log on the first process
        logger.info('Start training...')

    scaler = GradScaler(enabled=True)

    for epoch in range(start_epoch, cfg.TRAINING.NUM_EPOCHS):
        if global_rank != 0: # only log on the first process
            blockprint() 

        print('Epoch %d (%d/%s):' % (epoch + 1, epoch + 1, cfg.TRAINING.NUM_EPOCHS))
        logger.info('Epoch %d (%d/%s):' ,epoch + 1, epoch + 1, cfg.TRAINING.NUM_EPOCHS)
        print('lr=',optimizer.state_dict()['param_groups'][0]['lr'])

        train_total_loss = train(
            encoder=models["encoder"],
            depth_decoder=models["depth"],
            pose_encoder=models["pose_encoder"],
            pose=models["pose"],
            dataloader=train_iterator,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            global_rank=global_rank,
            cfg=cfg
        )

        torch.cuda.empty_cache() # clear cache between train and val

        val_total_loss = validate(
            encoder=models["encoder"],
            depth_decoder=models["depth"],
            pose_encoder=models["pose_encoder"],
            pose=models["pose"],
            dataloader=val_iterator,
            criterion=criterion,
            scaler=scaler,
            device=device,
            global_rank=global_rank,
            cfg=cfg
        )

        if global_rank == 0: # only log on the first process
            save_checkpoint(
                epoch + 1,
                models["encoder"],
                optimizer,
                str(checkpoints_dir),
                modelnet='encoder')
            save_checkpoint(
                epoch + 1,
                models["depth"],
                optimizer,
                str(checkpoints_dir),
                modelnet='depth')
            save_checkpoint(
                epoch + 1,
                models["pose_encoder"],
                optimizer,
                str(checkpoints_dir),
                modelnet='pose_encoder')
            save_checkpoint(
                epoch + 1,
                models["pose"],
                optimizer,
                str(checkpoints_dir),
                modelnet='pose')
            print('Saving model....')
            print(train_total_loss,val_total_loss)
            logger.info(f'Training Loss: Total: {train_total_loss:.2f}')
            logger.info(f'Validation Loss: Total: {val_total_loss:.2f}')


    if global_rank == 0: # only log on the first process
        logger.info('End of training...')
    

if __name__ == '__main__':
    arg = parse_args()
    cfg = load_cfg(arg.config_file)
    main(cfg)
