import torch
from escnn.group import SO3, GroupElement

G = SO3(maximum_frequency=6)
g = GroupElement(torch.zeros(3), group=G, param="ZYZ")
print(g)
g_irreps = [irrep(g) for irrep in G.irreps()]

euler_ang = torch.zeros(3)
euler_ang[1] = torch.pi/2
rot = GroupElement(euler_ang, group=G, param="ZYZ")
rotated = g @ rot
print(rotated.to("ZYZ"))
# print([irrep(rotated) for irrep in G.irreps()])
print()