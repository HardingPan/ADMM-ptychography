import  numpy as np


def overlap_rate(r,delta):
    # This formula is derived by the author himself
    overlap=(2*r**2*np.arctan(np.sqrt(r*r-delta**2/4)/(0.5*delta)) - delta*np.sqrt(r**2-0.25*delta**2)) / (3.14*r**2)
    print("overlap_rate is" , overlap)
    return overlap




