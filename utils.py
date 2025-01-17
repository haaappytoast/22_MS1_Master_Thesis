from typing import Tuple
import torch
import torch.nn.functional as F


@torch.jit.script
def rotatepoint(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q_v = [v[0], v[1], v[2], 0]
    # return quatmultiply(quatmultiply(q, q_v), quatconj(q))[:-1]
    #
    # https://fgiesen.wordpress.com/2019/02/09/rotating-a-single-vector-using-a-quaternion/
    q_r = q[...,3:4]
    q_xyz = q[...,:3]
    t = 2*torch.linalg.cross(q_xyz, v)
    return v + q_r * t + torch.linalg.cross(q_xyz, t)

@torch.jit.script
def heading_zup(q: torch.Tensor) -> torch.Tensor:
    ref_dir = torch.zeros_like(q[...,:3])
    ref_dir[..., 0] = 1                         # x-dir
    ref_dir = rotatepoint(q, ref_dir)
    return torch.atan2(ref_dir[...,1], ref_dir[...,0])

@torch.jit.script
def heading_yup(q: torch.Tensor) -> torch.Tensor:
    ref_dir = torch.zeros_like(q[...,:3])
    ref_dir[..., 0] = 1
    ref_dir = rotatepoint(q, ref_dir)
    return torch.atan2(-ref_dir[...,2], ref_dir[...,0])

@torch.jit.script
def quatnormalize(q: torch.Tensor) -> torch.Tensor:
    q = (1-2*(q[...,3:4]<0).to(q.dtype))*q
    return q / q.norm(p=2, dim=-1, keepdim=True)

@torch.jit.script
def quatmultiply(q0: torch.Tensor, q1: torch.Tensor):
    x0, y0, z0, w0 = torch.unbind(q0, -1)
    x1, y1, z1, w1 = torch.unbind(q1, -1)
    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    return quatnormalize(torch.stack((x, y, z, w), -1))

@torch.jit.script
def quatconj(q: torch.Tensor):
    return torch.cat((-q[...,:3], q[...,-1:]), dim=-1)

@torch.jit.script
def axang2quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    # axis: n x 3
    # angle: n
    theta = (angle / 2).unsqueeze(-1)
    axis = axis / (axis.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9))
    xyz = axis * torch.sin(theta)
    w = torch.cos(theta)
    return quatnormalize(torch.cat((xyz, w), -1))
 
@torch.jit.script
def quatdiff_normalized(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # quaternion representation of the rotation from unit vector a to b
    # need to check if a == -b
    # if a == -b: q = *a, 0         # 180 degree around any axis
    w = (a*b).sum(-1).add_(1)
    xyz = torch.linalg.cross(a, b)
    q = torch.cat((xyz, w.unsqueeze_(-1)), -1)
    return quatnormalize(q)

@torch.jit.script
def wrap2pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))

@torch.jit.script
def quat2axang(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    w = q[..., 3]

    sin = torch.sqrt(1 - w * w)
    mask = sin > 1e-5

    angle = 2 * torch.acos(w)
    angle = wrap2pi(angle)
    axis = q[..., 0:3] / sin.unsqueeze_(-1)

    z_axis = torch.zeros_like(axis)
    z_axis[..., -1] = 1

    angle = torch.where(mask, angle, z_axis[...,0])
    axis = torch.where(mask.unsqueeze_(-1), axis, z_axis)
    return axis, angle

@torch.jit.script
def quat2expmap(q: torch.Tensor) -> torch.Tensor:
    ax, ang = quat2axang(q)
    return ang.unsqueeze(-1)*ax

@torch.jit.script
def slerp(q0, q1, frac):
    c = q0[..., 3]*q1[..., 3] + q0[..., 0]*q1[..., 0] + \
        q0[..., 1]*q1[..., 1] + q0[..., 2]*q1[..., 2]
    q1 = torch.where(c.unsqueeze_(-1) < 0, -q1, q1)

    c = c.abs_()
    s = torch.sqrt(1.0 - c*c)
    t = torch.acos(c)

    c1 = torch.sin((1-frac)*t) / s
    c2 = torch.sin(frac*t) / s
    
    x = c1*q0[..., 0:1] + c2*q1[..., 0:1]
    y = c1*q0[..., 1:2] + c2*q1[..., 1:2]
    z = c1*q0[..., 2:3] + c2*q1[..., 2:3]
    w = c1*q0[..., 3:4] + c2*q1[..., 3:4]

    q = torch.cat((x, y, z, w), dim=-1)
    q = torch.where(s < 0.001, 0.5*q0+0.5*q1, q)
    q = torch.where(c >= 1, q0, q)
    return q


@torch.jit.script
def quat_conjugate(x):
    """
    quaternion with its imaginary part negated
    """
    return torch.cat([-x[..., :3], x[..., 3:]], dim=-1)

@torch.jit.script
def quat_inverse(x):
    """
    The inverse of the rotation
    """
    return quat_conjugate(x)

@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

@torch.jit.script
def quat_to_tan_norm(q):
    # type: (Tensor) -> Tensor
    # represents a rotation using the tangent and normal vectors
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1                         # x-dir
    tan = quat_rotate(q, ref_tan)
    
    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1                       # z-dir
    norm = quat_rotate(q, ref_norm)
    
    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan

@torch.jit.script
def tan_norm_to_rotmat(norm_tan):
    # type: (Tensor) -> Tensor
    tan = F.normalize(norm_tan[..., 0:3])
    norm = F.normalize(norm_tan[..., 3:6])
    binorm = F.normalize(torch.cross(norm_tan[..., 3:6], norm_tan[..., 0:3], dim= -1), dim=-1)
    
    rot_mat = torch.cat([tan, binorm, norm], dim = len(tan.shape)-1)
    return rot_mat

@torch.jit.script
def calc_heading_quat(q):
    # type: (Tensor) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = heading_zup(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1
    heading_q = axang2quat(axis, heading)
    return heading_q

def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
