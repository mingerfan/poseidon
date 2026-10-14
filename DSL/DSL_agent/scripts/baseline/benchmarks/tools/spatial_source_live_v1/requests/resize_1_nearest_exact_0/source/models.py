from fixtures import cases
from spatial_component_fixtures import contexts
def build():
    return next(x[1] for x in [*cases(),*contexts()] if x[0]=='resize_1_nearest_exact_0')
