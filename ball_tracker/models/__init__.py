from .hrnet import HRNet

__factory = {
    'hrnet': HRNet
        }

def build_model(cfg):
    model_name = cfg['model']['name']
    if not model_name in __factory.keys():
        raise KeyError('invalid model: {}'.format(model_name ))
    
    if model_name == 'hrnet':
        model = __factory[model_name](cfg['model'])
    else:
        raise NotImplementedError(f'Model {model_name} is not implemented in this ball_tracker version. Only hrnet is available.')

    return model

