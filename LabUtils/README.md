# LabUtils

This is a general purpose library written for personal use. Given it's large size and the already public nature of all the models, I am not providing it here. Please don't hesitate to ask by email or pull request if you need to see my version of it. 


Otherwise, the only purpose this library servers is to provide a Model object as follows : 

```
class ModelEncapsulation():

    def __init__(self,net,transforms=None,weights=None,did_we_train_head=False):
        self.model = net
        self.transforms = transforms_used_for_that_model
        self.local_weights = path/to/weights/stored/on/your/machine

        self.model.eval()

```

