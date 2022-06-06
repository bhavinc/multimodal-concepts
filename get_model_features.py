
import os
import torch
import pickle
from PIL import Image
from tqdm.auto import tqdm

from LabUtils.model_zoo import get_model, SUPPORTED_MODELS 




def main(model_name):

    model_instance = get_model(model_name,pretrained=True)
    transform_val = model_instance.transforms
    print (f"Getting weights from : {model_instance.local_weights}")

    net = model_instance.model
    net.eval()
    net.to(device)


    def get_fmri_test_representations(net,layers_to_be_hooked):

        activation = {}
        def get_activation(name):
            def hook(net, input, output):
                activation[name] = output.detach().clone().cpu().numpy()
            return hook

        for k,v in net.named_modules():
            if layers_to_be_hooked is not None:
                if k in layers_to_be_hooked:
                    v.register_forward_hook(get_activation(k))
            else : 
                v.register_forward_hook(get_activation(k))

        representation_dict = {}
        for cat_name in os.listdir(data_dir):
            if 'JPEG' in cat_name:
                
                
                print ()
                print (cat_name)
                print ('-'*20)

                img_path = os.path.join(data_dir,cat_name)
                image = Image.open(img_path).convert("RGB")
                tensor_image = transform_val(image).unsqueeze(0)

                if model_name == 'clip ViT-B/32':
                    tensor_image = tensor_image.half()

                activation = {}
                _preds = net(tensor_image.to(device))
                print (activation.keys())

                representation_dict[str(cat_name)] = activation

        return representation_dict 

    rpdict = get_fmri_test_representations(net,layers_to_be_hooked)
    return rpdict


if __name__ == '__main__':

    # model_name = 'resnet'
    # layers_to_be_hooked = ['fc','layer1.0']

    device = torch.device('cpu')
    data_dir = "/media/bhavin/My Passport/milad/bigbigan/fMRI/images/test" #get test images
    layers_to_be_hooked = None
    save_dir = '/mnt/HD2/bhavin/all_layerwise_reps_new'
 


    for _,model_name in enumerate(tqdm(SUPPORTED_MODELS)):
        print ('Getting model name : ',model_name)
        representations = main(model_name)


        # save the representations
        if save_dir is not None:
            fname = f'all_{model_name}_representations_fmri_test.p'
            with open(os.path.join(save_dir,fname),'wb') as f:
                pickle.dump(representations,f)
