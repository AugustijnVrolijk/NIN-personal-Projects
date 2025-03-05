import torch
from torch import nn
import torch.nn.functional as F



class testConv(nn.Module):
    def __init__(self):
        super(testConv, self).__init__()
        rows, cols = 5, 5
        temp = prepConvWeights(rows, cols)
        self.weight_tensor = torch.nn.Parameter(temp, requires_grad=False)
        temp2 = prepFeatureWeights()
        self.second_weight_tensor = torch.nn.Parameter(temp2, requires_grad=False)
        print(f"weight_tensor shape: {self.weight_tensor.shape} of type: {self.weight_tensor.dtype}")
        print(f"second_weight_tensor shape: {self.second_weight_tensor.shape} of type: {self.second_weight_tensor.dtype}")

        return

    def forward(self, input: torch.Tensor):
        postConv = F.conv2d(input,torch.abs(self.weight_tensor))
        self.valA = postConv
        postMult = torch.mul(postConv,self.second_weight_tensor)
        self.valB = postMult
        finalVals = torch.sum(postMult,-1,keepdim=True)
        return finalVals
    
def prepInput(num_matrices, rows, cols, verbose = False) -> torch.Tensor: 
    #creates a tensor of 1, 1, img, features


    # Create the base 5x5 matrix
    buff  = 11
    tensor0 = torch.zeros((5,5), dtype=torch.int64)
    tensor1 = torch.arange(buff, cols + buff) + torch.arange(rows).unsqueeze(1)
    tensor2 = torch.ones((5,5), dtype=torch.int64)

    # Extend to 3D by adding multiples of 10

    merged_tensor = torch.stack([tensor0, tensor1, tensor2], dim=0)
    tensor_4d = merged_tensor.unsqueeze(0)
    tensor_collapsed = tensor_4d.view(tensor_4d.shape[0],tensor_4d.shape[1],tensor_4d.shape[2]*tensor_4d.shape[3],1)
    tensor_reshaped = tensor_collapsed.permute(0,-1,2,1)

    if verbose:
        print(f"tensor_3d shape: {merged_tensor.shape}\n {merged_tensor}")
        print(f"tensor_4d shape: {tensor_4d.shape}\n {tensor_4d}")
        print(f"tensor_collapsed shape: {tensor_collapsed.shape}\n {tensor_collapsed}")
        print(f"tensor_reshaped shape: {tensor_reshaped.shape}\n {tensor_reshaped}")
    return tensor_reshaped

def prepConvWeights(rows, cols, verbose = False):
    #creates a tensor of nNeurons, 1, img, 1
    if rows != cols:
        print("error with prep Conv Weights")
        exit()
    tensorOnes = torch.ones(rows, dtype=torch.int64)
    tensorZeros = torch.zeros(rows, dtype=torch.int64)

    tensor1 = tensorOnes.repeat(5, 1)
    tensor2 = torch.stack([tensorZeros, tensorOnes, tensorZeros, tensorOnes, tensorZeros], dim=0)
    tensor3 = torch.arange(1,26).view(5,5)
    tensor3[3, :] = tensorZeros
    tensor3[0, :] = tensorOnes
    tensor4 = torch.full((5,5), 10)

    merged_tensor = torch.stack([tensor1, tensor2, tensor3, tensor4], dim=0)
    tensor_4d = merged_tensor.unsqueeze(0)
    tensor_collapsed = tensor_4d.view(tensor_4d.shape[0],tensor_4d.shape[1],tensor_4d.shape[2]*tensor_4d.shape[3],1)
    tensor_reshaped = tensor_collapsed.permute(1,0,2,-1)

    if verbose:
        print(f"tensor1 shape: {tensor1.shape}\n {tensor1}")
        print(f"tensor2 shape: {tensor2.shape}\n {tensor2}")
        print(f"tensor3 shape: {tensor3.shape}\n {tensor3}")
        print(f"tensor4 shape: {tensor4.shape}\n {tensor4}")
        print(f"merged_tensor shape: {merged_tensor.shape}\n {merged_tensor}")
        print(f"tensor_4d shape: {tensor_4d.shape}\n {tensor_4d}")
        print(f"tensor_collapsed shape: {tensor_collapsed.shape}\n {tensor_collapsed}")
        print(f"tensor_reshaped shape: {tensor_reshaped.shape}\n {tensor_reshaped}")
    return tensor_reshaped

def prepFeatureWeights(verbose = False):
    #creates a tensor of 1, nNeurons, 1, features
    # hardcode nNeurons and features to be 4 and 3 respectively
    tensor1 = torch.full((3,), 0)
    tensor2 = torch.full((3,), 1)
    tensor3 = torch.full((3,), 2)
    tensor4 = torch.full((3,), 10)
    merged_tensor = torch.stack([tensor1, tensor2, tensor3, tensor4], dim=0)
    tensor_3d = merged_tensor.unsqueeze(0)
    tensor_4d = tensor_3d.unsqueeze(0)
    tensor_reshaped = tensor_4d.permute(0,2,1,-1)

    if verbose:
        print(f"tensor1 shape: {tensor1.shape}\n {tensor1}")
        print(f"tensor2 shape: {tensor2.shape}\n {tensor2}")
        print(f"tensor3 shape: {tensor3.shape}\n {tensor3}")
        print(f"tensor4 shape: {tensor4.shape}\n {tensor4}")
        print(f"merged_tensor shape: {merged_tensor.shape}\n {merged_tensor}")
        print(f"tensor_3d shape: {tensor_3d.shape}\ntensor_4d shape:{tensor_4d.shape}")
        print(f"tensor_reshaped shape: {tensor_reshaped.shape}\n {tensor_reshaped}")

    return tensor_reshaped

def main():
    tester = testConv()
    num_matrices = 3  # Number of 5x5 matrices
    rows, cols = 5, 5
    tensor_reshaped = prepInput(num_matrices, rows, cols)
    print(f"tensor_reshaped shape: {tensor_reshaped.shape} of type: {tensor_reshaped.dtype}")
    
    finalVal = tester(tensor_reshaped)
    valA = tester.valA
    valB = tester.valB
    breakpoint()
    pass

if __name__ == "__main__":
    main()
        