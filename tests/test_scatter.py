import torch

from mecasnet.model import scatter_logsumexp


def test_scatter_logsumexp_matches_manual_groups() -> None:
    source = torch.tensor([0.0, 1.0, -2.0, 3.0])
    index = torch.tensor([0, 0, 1, 1])

    actual = scatter_logsumexp(source, index, dim_size=3)
    expected = torch.tensor([
        torch.logsumexp(source[:2], dim=0),
        torch.logsumexp(source[2:], dim=0),
        -torch.inf,
    ])

    torch.testing.assert_close(actual, expected)


def test_scatter_logsumexp_supports_vector_scores() -> None:
    source = torch.tensor([[0.0, 1.0], [1.0, 2.0], [-1.0, 4.0]])
    index = torch.tensor([0, 0, 1])

    actual = scatter_logsumexp(source, index, dim_size=2)

    torch.testing.assert_close(actual[0], torch.logsumexp(source[:2], dim=0))
    torch.testing.assert_close(actual[1], source[2])

