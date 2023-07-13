
import numpy as np
from scipy.optimize import quadratic_assignment, linear_sum_assignment
from scipy.spatial import distance_matrix
from collections import Counter
import random
def greed_assignment(A,B):
    num=A.shape[0]
    dM=distance_matrix(A,B)
    indexA=np.arange(num,dtype=int)
    indexB=np.zeros(num,dtype=int)
    dmax=dM.max()-dM.min()+1
    for i in range(num):
        d=np.argmin(dM)
        dA,dB=d//num,d%num
        indexB[dB]=indexA[dA]
        dM[dA,:]+=dmax
        dM[:,dB]+=dmax
    return indexB
def LAP(A,B):
    num=A.shape[0]
    dM=distance_matrix(B,A)
    row_ind, col_ind = linear_sum_assignment(dM)
    return col_ind
def solve_unbalanced_linear_assignment_bruteforce(cost_matrix, pad_value=1e6):
    """
    Solve the Unbalanced Linear Assignment Problem using the Hungarian algorithm.
    
    Args:
    - cost_matrix (torch.Tensor): A 2D tensor representing the cost matrix.
    - pad_value (float): Value used for padding the cost matrix.
    
    Returns:
    - assignment (list of tuples): A list of tuples representing the assigned rows and columns.
    - total_cost (torch.Tensor): The total cost of the assignment.
    """
    
    # Convert the cost matrix to a numpy array
    cost_matrix_np = cost_matrix.detach().clone().cpu().numpy()
    
    # Expand the cost matrix to be square if it's not
    num_rows, num_cols = cost_matrix_np.shape
    max_dim = max(num_rows, num_cols)
    padded_cost_matrix = np.full((max_dim, max_dim), pad_value)
    padded_cost_matrix[:num_rows, :num_cols] = cost_matrix_np
    
    # Solve the linear sum assignment problem on the padded matrix
    row_ind, col_ind = linear_sum_assignment(padded_cost_matrix)
    
    # Filter out the padding assignments and build the assignment list
    assignment = [(r, c) for r, c in zip(row_ind, col_ind) if r < num_rows and c < num_cols]
    
    # Calculate the total cost
    total_cost = cost_matrix[tuple(zip(*assignment))].sum()

    return assignment, total_cost

def ULA_loss(x,y):
    x=x.reshape(-1)
    y=y.reshape(-1)

    if len(x)>len(y):
        x,y=y,x
    #print(x,y)
    x,y=x.repeat(len(y),1).T,y.repeat(len(x),1)
    M=(x-y)**2
    _,loss=solve_unbalanced_linear_assignment_bruteforce(M)
    return loss/len(x)

def ULA_assign(x,y):
    x=x.reshape(-1)
    y=y.reshape(-1)

    if len(x)>len(y):
        x,y=y,x
    #print(x,y)
        x,y=x.repeat(len(y),1).T,y.repeat(len(x),1)
        M=(x-y)**2
        assignment,loss=solve_unbalanced_linear_assignment_bruteforce(M)
        
        out=np.ones(len(y))*(-1)
        out=out.long()
        for u in assignment:
            out[u[1]]=u[0]
        
        
    else:
        x,y=x.repeat(len(y),1).T,y.repeat(len(x),1)
        M=(x-y)**2
        assignment,loss=solve_unbalanced_linear_assignment_bruteforce(M)
        out=[u[1] for u in assignment]
    return out



def get_kth_most_frequent_elements(lst, k):
    counts = Counter(lst)
    most_common = counts.most_common(k)
    max_idx = [idx for idx, _ in most_common]
    phase_mask = [idx in max_idx for idx in lst]


    event_mask = [idx in max_idx for idx in range(max(list(counts.keys()))+1)]


    return phase_mask,event_mask


def remove_elements_with_probability(list_data, removal_probability):
    # Create a new list to store the elements that will not be removed
    new_list = []

    # Iterate over the elements in the original list
    for element in list_data:
        # Check if the element should be removed
        if random.random() <= removal_probability:
            continue  # Skip this element

        # Add the element to the new list
        new_list.append(element)

    return new_list