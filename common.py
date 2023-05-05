

def showmodel(model):
    print("="*20, "查看模型", "="*20)
    for name, param in model.named_parameters():
        print(f"{name} | {param.size()} | {param[:2]}")



def showmodel_short(model):
    print("="*20, "查看模型", "="*20)
    for name, param in model.named_parameters():
        print(f"{name} | {param.size()}")
