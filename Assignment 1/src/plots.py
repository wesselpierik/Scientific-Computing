from matplotlib import pyplot as plt

def Inch(d_cm):

    d_inch = d_cm / 2.54

    return d_inch

rcCustom_wide = plt.rcParams.copy()
rcCustom_wide["figure.dpi"] = 150
rcCustom_wide["figure.figsize"] = (Inch(28.58), Inch(12.09))

rcCustom = plt.rcParams.copy()
rcCustom["figure.dpi"] = 150
rcCustom["figure.figsize"] = (Inch(14.29), Inch(12.09))

