import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

def custom_summary(model, input_size, device="cuda", batch_size=-1):
    
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in ["cuda", "cpu"], "Invalid device specified: {}".format(device)
    
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Handle multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    
    # Create properties
    summary = OrderedDict()
    hooks = []

    # Register hooks
    model.apply(register_hook)

    # Make a forward pass
    model.to(device)
    x = [inp.to(device) for inp in x]
    model(*x)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Print summary
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    
    total_params = 0
    total_output = 0
    trainable_params = 0
    block_counter = 0  # Initialize block counter
    print("----------------------------------------------------------------")
    print("Block-0")
    print("----------------------------------------------------------------")
    
    for layer in summary:
        output_shape = summary[layer]["output_shape"]

        if "Block" in layer:
            block_counter += 1  # Increment block counter
            print("----------------------------------------------------------------")
            print(f"Block-{block_counter}")
            print("----------------------------------------------------------------")
        
        if isinstance(output_shape[0], list):
            output_shape = output_shape[0]  # Take the first element if it is a list
        
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(output_shape),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(output_shape)
        
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        
        print(line_new)

    # Assume 4 bytes/number (float on cuda)
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))

    total_size = total_params_size + total_output_size + total_input_size

    # Print total summary
    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary
