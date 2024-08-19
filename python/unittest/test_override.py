import torch
import ark


# Test the implementation
def test_override():

    # inputs
    a = torch.tensor([1.0, 2.0],requires_grad=True)
    b = torch.tensor([3.0, 4.0],requires_grad=True)

    # Outputs of operations computed by ARK
    # c = matmul(a,b)
    c = torch.matmul(a,b)
    # d = relu(c)
    d = torch.relu(c)
        
    
    ark._OperationRegistry.tns_map = {
        id(a): a,
        id(b): b,
        id(c) : c,
        id(d) : d,
    }

    # Recorded the performed operations
    ark._OperationRegistry.record_op([id(a), id(b)], [id(c)], torch.matmul)
    ark._OperationRegistry.record_op([id(c)], [id(d)], torch.nn.functional.relu)

    #Execute precomputed operations
    #ark._OperationRegistry.execute_overridden_ops()
    
    loss = c + d + a.sum() + b.sum()
    loss.backward()

    print("\nResults:")
    print("a.grad:", a.grad)
    print("b.grad:", b.grad)
    print("c.grad:", c.grad)
    print("d.grad:", d.grad)

    # Verify results
    assert a.grad is not None, "Gradient for 'a' should not be None"
    assert b.grad is not None, "Gradient for 'b' should not be None"

    print("\nAll tests passed!")

test_override()