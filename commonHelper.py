import torch


def get_device(use_seed = True):
    is_gpu_available = torch.cuda.is_available()
    
    SEED = 1
    device = "cpu"
    if(is_gpu_available):
        device = "cuda"
        # This ensures that computations involving randomness on the GPU will produce the same results
        # when the seed is the same, even if you run the code multiple times.
        if(use_seed):
            torch.cuda.manual_seed(seed=SEED)
    
    return device


def get_correct_prediction_count(predictions : torch.Tensor, valid_labels):
    # torch.argmax locates the indices of the maximum elements within a tensor.
    # It does this along a specified dimension (axis).
    return predictions.argmax(dim=1).eq(valid_labels).sum().item()

def get_true_and_false_indices(predictions : torch.Tensor, valid_labels):

    comparison_result = predictions.argmax(dim=1).eq(valid_labels)
    # Get the indices of True values
    true_indices = torch.where(comparison_result)[0]
    false_indices = torch.where(~comparison_result)[0]
    return true_indices, false_indices

def print_analysis_shapes(batch_data, batch_labels, batch_predictions):
        argmax_data = batch_predictions.argmax(dim=1)
        true_indices, false_indices = get_true_and_false_indices(batch_predictions, batch_labels)

        print(f" Shape of data : {batch_data.shape}, label: {batch_labels.shape}")
        print(f"Shape of predicted data : {batch_predictions.shape}")
        print(f"labels : {batch_labels}")
        print("-----------------------------------------------------")
        print(f"argmax_data : {argmax_data}")

        print(f"true_indices : {true_indices}")
        print(f"false_indices : {false_indices}")

