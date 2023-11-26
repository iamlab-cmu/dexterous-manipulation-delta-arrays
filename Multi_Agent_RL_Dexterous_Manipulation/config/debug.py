from stl import mesh
import numpy as np

mesh = mesh.Mesh.from_file('./assets/block.stl')

vol, cog, inertia = mesh.get_mass_properties()
print("Volume                                  = {0}".format(vol))
print("Position of the center of gravity (COG) = {0}".format(cog))
print("Inertia matrix at expressed at the COG  = {0}".format(inertia[0,:]))
print("                                          {0}".format(inertia[1,:]))
print("                                          {0}".format(inertia[2,:]))