import rich
import torch


def test(x: torch.Tensor, y: torch.Tensor):
    return x.select_scatter(y, 0, -1)


rich.print(test(torch.tensor([0.0, 1.0]), torch.tensor(3.0)))
fn = torch.jit.trace(test, (torch.tensor([0.0, 1.0]), torch.tensor(3.0)))
rich.print(fn.graph)
torch.jit.save(fn, "test.pt")
rich.print(torch.jit.load("test.pt")(torch.tensor([0.0, 1.0]), torch.tensor(3.0)))
