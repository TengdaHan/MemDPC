from .resnet_2d3d import * 

def select_resnet(network,):
    param = {'feature_size': 1024}
    if network == 'resnet18':
        model = resnet18_2d3d_full(track_running_stats=True)
        param['feature_size'] = 256
    elif network == 'resnet34':
        model = resnet34_2d3d_full(track_running_stats=True)
        param['feature_size'] = 256 
    elif network == 'resnet50':
        model = resnet50_2d3d_full(track_running_stats=True)
    else: 
        raise NotImplementedError

    return model, param