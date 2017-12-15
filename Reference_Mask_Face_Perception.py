from nltools.mask import create_sphere

ROFA= create_sphere([43,-78,-13],radius=5)
LOFA= create_sphere([-41,-80,-12],radius=5)

RpFFA=create_sphere([42,-52,-20],radius=5)
LpFFA=create_sphere([-40,-54,-20],radius=5)

RaFFA=create_sphere([43,-24,-25],radius=5)
LaFFA=create_sphere([-42,-26,-23],radius=5)

RpcSTS=create_sphere([55,-59,7],radius=5)
LpcSTS=create_sphere([-57,-62,9],radius=5)

RpSTS=create_sphere([54,-38,4],radius=5)
LpSTS=create_sphere([-58,-41,4],radius=5)

RaSTS=create_sphere([55,-7,-15],radius=5)
LaSTS=create_sphere([-58,-6,-16],radius=5)

from nilearn.plotting import plot_roi,show
plot_roi(ROFA,title='ROFA')
plot_roi(LOFA,title='LOFA')

plot_roi(RpFFA,title='RpFFA')
plot_roi(LpFFA,title='LpFFA')

plot_roi(RaFFA,title='RaFFA')
plot_roi(LaFFA,title='LaFFA')

plot_roi(RpcSTS,title='RpcSTS')
plot_roi(LpcSTS,title='LpcSTS')
plot_roi(RpSTS,title='RpSTS')
plot_roi(LpSTS,title='LpSTS')
plot_roi(RaSTS,title='RaSTS')
plot_roi(LaSTS,title='LaSTS')

show()