import torch

from uvnet_retrieval.geometry_targets import compute_geometry_targets
from uvnet_retrieval.heads import GeometryHead
from uvnet_retrieval.losses import masked_mse
from uvnet_retrieval.masking import apply_node_mask
from uvnet_retrieval.uvnet_wrapper import UVNetEncoder


def compute_losses(
    graph,
    mask,
    student=None,
    teacher=None,
    geom_head=None,
    device="cpu",
    geom_weight=0.1,
):
    if student is None:
        student = UVNetEncoder().to(device)
    if teacher is None:
        teacher = UVNetEncoder().to(device)
        teacher.load_state_dict(student.state_dict())
        for p in teacher.parameters():
            p.requires_grad_(False)

    face_feat = graph.ndata["x"].to(device)
    mask = mask.to(device)
    masked_feat = apply_node_mask(face_feat.clone(), mask)
    graph.ndata["x"] = masked_feat
    student_tokens, _ = student(graph)
    with torch.no_grad():
        graph.ndata["x"] = face_feat
        teacher_tokens, _ = teacher(graph)

    loss_c = masked_mse(student_tokens, teacher_tokens, mask)
    if geom_head is None:
        geom_head = GeometryHead(student_tokens.size(-1)).to(device)
    geom_pred = geom_head(student_tokens[mask])
    geom_target = compute_geometry_targets(face_feat)[mask]
    loss_a = torch.nn.functional.mse_loss(geom_pred, geom_target)
    graph.ndata["x"] = face_feat
    return loss_c + geom_weight * loss_a


def main():
    print("MAE training loop placeholder")

if __name__ == "__main__":
    main()
